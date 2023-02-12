module BareVertexPolar

using GaussLegendrePolynomial
using DSEMathWrapper
using NLsolve, Dierckx
using HDF5
using GluonModel
const N_kernel = 5::Integer
  

include(joinpath(@__DIR__,"CommonFunction","AllocateCommonVariables.jl"))
include(joinpath(@__DIR__,"CommonFunction","AssignInitValue.jl"))
include(joinpath(@__DIR__,"CommonFunction","TransformABC.jl"))
include(joinpath(@__DIR__,"CommonFunction","InterpolateABC.jl"))
include(joinpath(@__DIR__,"..","..","Constants.jl"))

function solve(::Type{T},constants;result_file::String) where{T<:Real}
  solve(T, constants.mass, constants.mu; 
                constants.init_type,constants.gluon_model,
                constants.Gluon_D_Long, constants.Gluon_D_Tran, constants.Gluon_omega, constants.LambdaR, constants.LambdaV, 
                constants.N_PP, constants.N_QQ, constants.Np_PSI, constants.N_PHI,
                result_file)
end

function solve(::Type{T}, Mass::T, mu::T; 
                init_type,gluon_model,
                Gluon_D_Long::T, Gluon_D_Tran::T, Gluon_omega::T, LambdaR::T, LambdaV::T, 
                N_PP, N_QQ, Np_PSI, N_PHI,
                result_file::String) where {T<:AbstractFloat}
  
  Nq_PSI,
  pp,wpp,qq,wqq,
  psi_p,wpsi_p,psi_q,wpsi_q,
  zphi,wphi,
  Ap,Bp,Cp,
  Aq,Bq,Cq,ABC_init=AllocateCommonVariables_Polar(N_PP, N_QQ, Np_PSI, N_PHI,LambdaR,LambdaV)

  AssignInitValue!(init_type, Mass, N_PP, Np_PSI, Ap, Bp, Cp,ABC_init)

  Gluon_Long_T0Mu, Gluon_Tran_T0Mu = GluonModel.InitGluonModel(gluon_model,Gluon_D_Long,Gluon_D_Tran,Gluon_omega)

  @time AKernelAq, AKernelCq, 
  BKernelBq, 
  CKernelAq, CKernelCq=Init_Matrix_RL(mu, N_PP, N_QQ, Np_PSI, Nq_PSI, N_PHI,
                        pp, qq, psi_p, psi_q, zphi, wphi, Gluon_Tran_T0Mu, Gluon_Long_T0Mu)

  kernel(FUNC,ABC)=gap_equation_RL_T0Mu_polar!(T,ABC,FUNC,mu,Mass,
            N_PP,Np_PSI, Ap,Bp,Cp,
            N_QQ,Nq_PSI, Aq,Bq,Cq,
            AKernelAq,AKernelCq,BKernelBq,CKernelAq,CKernelCq,
            pp,wpp,qq,wqq,
            psi_p,wpsi_p,psi_q,wpsi_q)
    
  
  ABC=reinterpret(T,ABC_init) # ABC_init is a complex array, we have to convert it into real array to feed it into non-linear equation solver.
  @time result=nlsolve(kernel,ABC,show_trace=true,method=:anderson,ftol=1e-7)
  ABC=reinterpret(Complex{T},result.zero)
  SplitABC!(N_PP,Np_PSI,ABC,Ap,Bp,Cp)

  if result_file != "" 
    file=h5open(result_file,"w")
    write(file,"QuarkPropagator/A",Ap)
    write(file,"QuarkPropagator/B",Bp)
    write(file,"QuarkPropagator/C",Cp)
    close(file)
  end
  print("A[1,1]=", Ap[1,1]," B[1,1]=",Bp[1,1])

end

function Inte_outofz_RL_T0Mu!(Mu::T, N_PHI::Integer, vangle,
                              P,Q,PSI_p,PSI_q,
                              zphi_array, wphi_array, 
                              Gluon_Tran_T0Mu, Gluon_Long_T0Mu) where {T<:AbstractFloat}
    p4=P*PSI_p+im*Mu
    q4=Q*PSI_q+im*Mu
    pv=P*sqrt(1-PSI_p*PSI_p)
    qv=Q*sqrt(1-PSI_q*PSI_q)
  
    array = zeros(Complex{T}, N_kernel,N_PHI)
    for i = 1: N_PHI
      z_phi = zphi_array[i]
      k2_v = pv*pv + qv*qv - 2 * pv*qv*z_phi
      pq = pv*qv*z_phi
      pk = pv*pv - pv*qv*z_phi
      qk = pv*qv*z_phi - qv*qv
      k4 = real(p4-q4)
      k2 = k2_v + k4*k4
    
      GluonZT=Gluon_Tran_T0Mu(k2,Mu)
      GluonZL=Gluon_Long_T0Mu(k2,Mu)
    
      array[1,i] = (pq*GluonZL + 2.0*pk*qk / k2*GluonZL + 2.0*pk*qk / k2_v*GluonZT - 2.0*pk*qk / k2_v*GluonZL) 
      array[2,i] = GluonZL*(2.0*q4*k4*pk / k2)
      array[3,i] = 2.0*GluonZT + GluonZL
      array[4,i] = GluonZL*(2.0*p4*k4*qk / k2)
      array[5,i] = p4*q4*(2.0*GluonZT - GluonZL + 2.0*k4*k4 / k2*GluonZL)
    end
  
    CommonPart = 4.0 / 3.0 / (2 * pi)^3 * sqrt(1-PSI_q*PSI_q)*Q*Q*Q;
  
    for i=1:N_kernel
      vangle[i] = CommonPart * inte_array_cmplx(view(array,i,:), wphi_array)
    end
  end

function Init_Matrix_RL(Mu::T, N_PP, N_QQ, Np_PSI, Nq_PSI, N_PHI,
  pp_array, qq_array, psi_p_array, psi_q_array, zphi_array, wphi_array,
  Gluon_Tran_T0Mu, Gluon_Long_T0Mu) where {T<:AbstractFloat}

  AKernelAq=zeros(Complex{T},N_PP,N_QQ,Np_PSI,Nq_PSI)
  AKernelCq=zeros(Complex{T},N_PP,N_QQ,Np_PSI,Nq_PSI)
  BKernelBq=zeros(Complex{T},N_PP,N_QQ,Np_PSI,Nq_PSI)
  CKernelAq=zeros(Complex{T},N_PP,N_QQ,Np_PSI,Nq_PSI)
  CKernelCq=zeros(Complex{T},N_PP,N_QQ,Np_PSI,Nq_PSI)
  
  Threads.@threads for j2 = 1: Nq_PSI
    for i2 = 1: Np_PSI, j1 = 1: N_QQ, i1 = 1: N_PP
      P     = pp_array[i1]
      Q     = qq_array[j1]
      PSI_p = psi_p_array[i2]
      PSI_q = psi_q_array[j2]
      
      vangle = zeros(Complex{T},N_kernel)

      Inte_outofz_RL_T0Mu!(Mu,N_PHI, vangle,P,Q,PSI_p,PSI_q,
                              zphi_array, wphi_array, 
                              Gluon_Tran_T0Mu, Gluon_Long_T0Mu)
      
      AKernelAq[i1,j1,i2,j2]=vangle[1]
      AKernelCq[i1,j1,i2,j2]=vangle[2]
      BKernelBq[i1,j1,i2,j2]=vangle[3]
      CKernelAq[i1,j1,i2,j2]=vangle[4]
      CKernelCq[i1,j1,i2,j2]=vangle[5]
    end
  end
  return AKernelAq, AKernelCq, BKernelBq, CKernelAq, CKernelCq
end
  
  
"""
This is the kernel of the integration equations.
The input variables:
  Ap,Bp,Cp,
  Aq_matrix,Bq_matrix,Cq_matrix,
are matrices for the propagator.
However, you don't need to provide their values.
They are only here to avoid allocating spaces for every iteration.
"""
function gap_equation_RL_T0Mu_polar!(::Type{T},ABC_real,FUNC_real,Mu,Mass,
          N_PP,Np_PSI, Ap,Bp,Cp,
          N_QQ,Nq_PSI, Aq_matrix,Bq_matrix,Cq_matrix,
          AKernelAq,AKernelCq,BKernelBq,CKernelAq,CKernelCq,
          pp_array,wpp_array,qq_array,wqq_array,
          psi_p_array,wpsi_p_array,psi_q_array,wpsi_q_array)where{T}

  ABC =reinterpret(Complex{T},ABC_real)
  FUNC=reinterpret(Complex{T},FUNC_real)

  SplitABC!(N_PP,Np_PSI,ABC,Ap,Bp,Cp)

  InterpolateABC!(N_PP, N_QQ, Np_PSI, Nq_PSI, pp_array, qq_array, Ap, Bp, Cp, Aq_matrix, Bq_matrix, Cq_matrix)

  Threads.@threads for i1 = 1: N_PP
    for i2 = 1: Np_PSI
      P=pp_array[i1]
      Psi_p=psi_p_array[i2]
    
      p4=P*Psi_p+im*Mu;
      pv=P*sqrt(1-Psi_p*Psi_p);
    
      sum_mA = (0.0+0.0im)
      sum_mB = (0.0+0.0im)
      sum_mC = (0.0+0.0im)
      KerA = zeros(Complex{T},Nq_PSI)
      KerB = zeros(Complex{T},Nq_PSI)
      KerC = zeros(Complex{T},Nq_PSI)
    
      for j1 = 1: N_QQ
        for j2 = 1: Nq_PSI
          Aq=Aq_matrix[j1,j2];
          Bq=Bq_matrix[j1,j2];
          Cq=Cq_matrix[j1,j2];
        
        
          Q=qq_array[j1]
          Psi_q=psi_q_array[j2]
        
          q4=Q*Psi_q+im*Mu;
          qv=Q*sqrt(1-Psi_q*Psi_q);
        
          Denorm=1.0/(qv*qv*Aq*Aq+q4*q4*Cq*Cq+Bq*Bq);
        
          KerA[j2]= Denorm * (AKernelAq[i1,j1,i2,j2]*Aq + AKernelCq[i1,j1,i2,j2]*Cq);
          KerB[j2]= Denorm * (BKernelBq[i1,j1,i2,j2]*Bq);
          KerC[j2]= Denorm * (CKernelAq[i1,j1,i2,j2]*Aq + CKernelCq[i1,j1,i2,j2]*Cq);
        
        end
        sum_mA += wqq_array[j1] * inte_array_cmplx(KerA, wpsi_q_array)/(pv*pv);
        sum_mB += wqq_array[j1] * inte_array_cmplx(KerB, wpsi_q_array);
        sum_mC += wqq_array[j1] * inte_array_cmplx(KerC, wpsi_q_array)/(p4*p4);
      end

      FUNC[(i1-1) * 3 * Np_PSI + i2]              = Ap[i1,i2] - 1    - sum_mA;
      FUNC[(i1-1) * 3 * Np_PSI + Np_PSI + i2]     = Bp[i1,i2] - Mass - sum_mB;
      FUNC[(i1-1) * 3 * Np_PSI + 2 * Np_PSI + i2] = Cp[i1,i2] - 1    - sum_mC;
    end
  end
end

end # end module