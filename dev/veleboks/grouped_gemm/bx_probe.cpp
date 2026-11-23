#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <map>
#include <numeric>

// Diagnostic only: identical BX epilogue, different cuBLAS batching/layouts.
struct BXProbe {
  struct Groups {
    std::vector<int> m,n,k,la,lb,lc,count;
    torch::Tensor pointers;
    std::vector<const void*> left,right;
    std::vector<void*> out;
  };
  torch::Tensor x,b,y;
  std::vector<int> widths;
  Groups individual,buckets;
  bool column_major;
  int pack,rows,cols,version;
  struct LtCase {
    cublasLtMatmulDesc_t desc=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;
    std::vector<cublasLtMatmulAlgo_t> algorithms;
  };
  cublasLtHandle_t lt=nullptr;
  std::map<int,LtCase> lt_cases;
  struct LtRun {int first,count,gap; LtCase c;};
  LtCase lt_full;
  std::vector<LtRun> lt_runs;
  torch::Tensor workspace;

  BXProbe(torch::Tensor input,torch::Tensor gram,std::vector<int> sizes,bool column)
    : x(input),b(gram),widths(sizes),column_major(column) {
    TORCH_CHECK(x.is_cuda() && b.device()==x.device() && x.dim()==3,
                "CUDA tensors required");
    TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && b.scalar_type()==torch::kBFloat16,
                "BF16 tensors required");
    c10::cuda::CUDAGuard guard(x.device());
    pack=x.size(0); rows=x.size(1); cols=x.size(2);
    TORCH_CHECK(int(widths.size())==pack && b.sizes()==torch::IntArrayRef({pack,rows,rows}),
                "shape mismatch");
    TORCH_CHECK(x.stride(0)==int64_t(rows)*cols &&
                x.stride(1)==(column?1:cols) && x.stride(2)==(column?rows:1) &&
                b.is_contiguous(),"unsupported strides");
    y=torch::empty_strided(x.sizes(),x.strides(),x.options());
    y.zero_();
    std::vector<int> order(pack);
    std::iota(order.begin(),order.end(),0);
    individual=build(order,false);
    std::stable_sort(order.begin(),order.end(),[&](int i,int j){return widths[i]<widths[j];});
    buckets=build(order,true);
    cublasGetVersion(at::cuda::getCurrentCUDABlasHandle(),&version);
    TORCH_CHECK(cublasLtCreate(&lt)==CUBLAS_STATUS_SUCCESS,"Lt handle failed");
    workspace=torch::empty({64*1024*1024},x.options().dtype(torch::kUInt8));
    for(int w:widths) if(!lt_cases.count(w)) make_lt(w);
    fill_lt(lt_full,rows,pack,1);
    std::map<int,std::vector<int>> by_width;
    for(int i=0;i<pack;i++) by_width[widths[i]].push_back(i);
    for(auto& item:by_width) {
      auto& ids=item.second;
      for(size_t start=0;start<ids.size();) {
        int gap=start+1<ids.size()?ids[start+1]-ids[start]:1;
        size_t end=std::min(start+2,ids.size());
        while(end<ids.size() && ids[end]-ids[end-1]==gap) end++;
        lt_runs.push_back({ids[start],int(end-start),gap,{}});
        fill_lt(lt_runs.back().c,item.first,int(end-start),gap);
        start=end;
      }
    }
  }
  static void destroy_lt(LtCase& c) {
    if(c.desc) cublasLtMatmulDescDestroy(c.desc);
    if(c.a) cublasLtMatrixLayoutDestroy(c.a);
    if(c.b) cublasLtMatrixLayoutDestroy(c.b);
    if(c.c) cublasLtMatrixLayoutDestroy(c.c);
    if(c.d) cublasLtMatrixLayoutDestroy(c.d);
  }
  ~BXProbe() {
    for(auto& item:lt_cases) destroy_lt(item.second);
    destroy_lt(lt_full);
    for(auto& run:lt_runs) destroy_lt(run.c);
    if(lt) cublasLtDestroy(lt);
  }
  void make_lt(int w) {
    fill_lt(lt_cases[w],w,1,1);
  }
  void fill_lt(LtCase& c,int w,int batch,int gap) {
    int m=column_major?w:cols,n=column_major?cols:w,k=w;
    auto op=column_major?CUBLAS_OP_T:CUBLAS_OP_N;
    TORCH_CHECK(cublasLtMatmulDescCreate(&c.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS,"Lt desc failed");
    cublasLtMatmulDescSetAttribute(c.desc,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op));
    cublasLtMatrixLayoutCreate(&c.a,CUDA_R_16BF,column_major?k:m,column_major?m:k,column_major?rows:cols);
    cublasLtMatrixLayoutCreate(&c.b,CUDA_R_16BF,k,n,rows);
    cublasLtMatrixLayoutCreate(&c.c,CUDA_R_16BF,m,n,column_major?rows:cols);
    cublasLtMatrixLayoutCreate(&c.d,CUDA_R_16BF,m,n,column_major?rows:cols);
    int64_t stride_x=int64_t(gap)*rows*cols,stride_b=int64_t(gap)*rows*rows;
    for(auto layout:{c.a,c.b,c.c,c.d})
      cublasLtMatrixLayoutSetAttribute(layout,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch));
    int64_t sa=column_major?stride_b:stride_x,sb=column_major?stride_x:stride_b;
    cublasLtMatrixLayoutSetAttribute(c.a,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sa,sizeof(sa));
    cublasLtMatrixLayoutSetAttribute(c.b,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sb,sizeof(sb));
    cublasLtMatrixLayoutSetAttribute(c.c,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_x,sizeof(stride_x));
    cublasLtMatrixLayoutSetAttribute(c.d,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_x,sizeof(stride_x));
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t bytes=workspace.numel();
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
    cublasLtMatmulHeuristicResult_t results[8]; int returned=0;
    auto status=cublasLtMatmulAlgoGetHeuristic(lt,c.desc,c.a,c.b,c.c,c.d,pref,8,results,&returned);
    if(status==CUBLAS_STATUS_SUCCESS) for(int i=0;i<returned;i++)
      if(results[i].state==CUBLAS_STATUS_SUCCESS) c.algorithms.push_back(results[i].algo);
    cublasLtMatmulPreferenceDestroy(pref);
  }
  cublasStatus_t call_lt(LtCase& c,int i,int candidate) {
    TORCH_CHECK(candidate<int(c.algorithms.size()),"Lt heuristic unavailable");
    float alpha=1,beta=3.4445f;
    auto px=x.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols;
    return cublasLtMatmul(lt,c.desc,&alpha,individual.left[i],c.a,
      individual.right[i],c.b,&beta,px,c.c,individual.out[i],c.d,
      &c.algorithms[candidate],workspace.data_ptr(),workspace.numel(),
      c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
  }
  Groups build(const std::vector<int>& order,bool coalesce) {
    Groups g;
    auto host=torch::empty({3*pack},torch::TensorOptions().dtype(torch::kInt64));
    auto ptr=host.data_ptr<int64_t>();
    int previous=-1;
    for(int pos=0;pos<pack;pos++) {
      int i=order[pos],w=widths[i];
      TORCH_CHECK(w>0 && w<=rows,"invalid width");
      if(coalesce && w==previous) g.count.back()++;
      else {
        g.m.push_back(column_major?w:cols); g.n.push_back(column_major?cols:w);
        g.k.push_back(w); g.la.push_back(column_major?rows:cols);
        g.lb.push_back(rows); g.lc.push_back(column_major?rows:cols); g.count.push_back(1);
      }
      previous=w;
      auto px=x.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols;
      auto pb=b.data_ptr<at::BFloat16>()+int64_t(i)*rows*rows;
      auto py=y.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols;
      g.left.push_back(column_major?static_cast<const void*>(pb):static_cast<const void*>(px));
      g.right.push_back(column_major?static_cast<const void*>(px):static_cast<const void*>(pb));
      g.out.push_back(py);
      ptr[pos]=reinterpret_cast<int64_t>(g.left.back());
      ptr[pack+pos]=reinterpret_cast<int64_t>(g.right.back());
      ptr[2*pack+pos]=reinterpret_cast<int64_t>(py);
    }
    g.pointers=host.to(x.device());
    return g;
  }
  torch::Tensor run(std::string mode,int algo=-1) {
    c10::cuda::CUDAGuard guard(x.device());
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t pm;
    cublasGetPointerMode(handle,&pm);
    TORCH_CHECK(pm==CUBLAS_POINTER_MODE_HOST,"host pointer mode required");
    // Lt takes C=X, D=Y and avoids the full target copy.
    if(mode.rfind("lt",0)!=0) y.copy_(x);
    float alpha=1,beta=3.4445f;
    auto op=column_major?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
    if(mode=="lt_full") {
      status=call_lt(lt_full,0,algo<0?0:algo);
    } else if(mode=="lt_runs") {
      for(auto& run:lt_runs) {
        status=call_lt(run.c,run.first,algo<0?0:algo);
        if(status!=CUBLAS_STATUS_SUCCESS) break;
      }
    } else if(mode=="lt") {
      int candidate=algo<0?0:algo;
      for(int i=0;i<pack;i++) {
        auto& c=lt_cases[widths[i]];
        status=call_lt(c,i,candidate);
        if(status!=CUBLAS_STATUS_SUCCESS) break;
      }
    } else if(mode=="grouped" || mode=="coalesced") {
      auto& g=mode=="grouped"?individual:buckets;
      int n=g.count.size();
      std::vector<cublasOperation_t> opa(n,op),opb(n,CUBLAS_OP_N);
      std::vector<float> al(n,alpha),be(n,beta);
      auto p=g.pointers.data_ptr<int64_t>();
      status=cublasGemmGroupedBatchedEx(handle,opa.data(),opb.data(),g.m.data(),g.n.data(),
        g.k.data(),al.data(),reinterpret_cast<const void* const*>(p),CUDA_R_16BF,g.la.data(),
        reinterpret_cast<const void* const*>(p+pack),CUDA_R_16BF,g.lb.data(),be.data(),
        reinterpret_cast<void* const*>(p+2*pack),CUDA_R_16BF,g.lc.data(),n,g.count.data(),CUBLAS_COMPUTE_32F);
    } else if(mode=="batched") {
      auto& g=buckets;
      auto p=g.pointers.data_ptr<int64_t>();
      int offset=0;
      for(size_t i=0;i<g.count.size();i++) {
        status=cublasGemmBatchedEx(handle,op,CUBLAS_OP_N,g.m[i],g.n[i],g.k[i],&alpha,
          reinterpret_cast<const void* const*>(p+offset),CUDA_R_16BF,g.la[i],
          reinterpret_cast<const void* const*>(p+pack+offset),CUDA_R_16BF,g.lb[i],&beta,
          reinterpret_cast<void* const*>(p+2*pack+offset),CUDA_R_16BF,g.lc[i],g.count[i],
          CUBLAS_COMPUTE_32F,static_cast<cublasGemmAlgo_t>(algo));
        if(status!=CUBLAS_STATUS_SUCCESS) break;
        offset+=g.count[i];
      }
    } else if(mode=="single") {
      auto& g=individual;
      for(int i=0;i<pack;i++) {
        status=cublasGemmEx(handle,op,CUBLAS_OP_N,g.m[i],g.n[i],g.k[i],&alpha,
          g.left[i],CUDA_R_16BF,g.la[i],g.right[i],CUDA_R_16BF,g.lb[i],&beta,
          g.out[i],CUDA_R_16BF,g.lc[i],CUBLAS_COMPUTE_32F,static_cast<cublasGemmAlgo_t>(algo));
        if(status!=CUBLAS_STATUS_SUCCESS) break;
      }
    } else TORCH_CHECK(false,"unknown mode");
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"cuBLAS status=",int(status)," mode=",mode," algo=",algo);
    return y;
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<BXProbe>(m,"BXProbe")
    .def(pybind11::init<torch::Tensor,torch::Tensor,std::vector<int>,bool>())
    .def("run",&BXProbe::run,pybind11::arg("mode"),pybind11::arg("algo")=-1)
    .def_readonly("version",&BXProbe::version);
}
