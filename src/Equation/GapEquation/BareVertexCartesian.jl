module BareVertexCartesian

using GaussLegendrePolynomial
using NLsolve, Dierckx
using HDF5
using GluonModel
using Revise

const N_kernel = 5::Integer

include(joinpath(@__DIR__,"CommonFunction","AllocateCommonVariables.jl"))
include(joinpath(@__DIR__,"CommonFunction","AssignInitValue.jl"))
include(joinpath(@__DIR__,"CommonFunction","TransformABC.jl"))
include(joinpath(@__DIR__,"CommonFunction","InterpolateABC.jl"))
include(joinpath(@__DIR__,"..","..","Constants.jl"))

function inte_array_cmplx(func, w) 
  s = zero(eltype(func))
  for i = eachindex(func)
    s += func[i] * w[i]
  end
  return s
end

function solve(::Type{T},constants;result_file::String) where{T<:Real}
  solve(T, constants.mass, constants.mu; 
                constants.init_type,constants.gluon_model,
                constants.Gluon_D_Long, constants.Gluon_D_Tran, constants.Gluon_omega, constants.LambdaR_P4, constants.LambdaV_P4, 
                constants.LambdaR_PV, constants.LambdaV_PV, 
                constants.N_PV, constants.N_QV, constants.N_P4, constants.N_PHI,
                result_file)
end

function solve(::Type{T}, Mass::T, mu::T; 
                init_type,gluon_model,
                Gluon_D_Long::T, Gluon_D_Tran::T, Gluon_omega::T, LambdaR_P4::T, LambdaV_P4::T, 
                LambdaR_PV::T, LambdaV_PV::T, 
                N_PV, N_QV, N_P4, N_PHI,
                result_file::String) where {T<:AbstractFloat}

  N_Q4,
  pv,wpv,qv,wqv,
  p4,wp4,q4,wq4,
  zphi,wphi,
  Ap,Bp,Cp,Aq,Bq,Cq,ABC_init =AllocateCommonVariables_Cartesian(N_PV,N_QV,N_P4,N_PHI,LambdaR_P4,LambdaV_P4,LambdaR_PV,LambdaV_PV)

  AssignInitValue!(init_type, Mass, N_PV, N_P4, Ap, Bp, Cp,ABC_init)


  Gluon_Long_T0Mu, Gluon_Tran_T0Mu = GluonModel.InitGluonModel(gluon_model,Gluon_D_Long,Gluon_D_Tran,Gluon_omega)

  @time AKernelAq, AKernelCq, BKernelBq, CKernelAq, CKernelCq = 
              Init_Matrix_RL(mu, N_PV, N_QV, N_P4, N_Q4, N_PHI,
                              pv, qv, p4, q4, zphi, wphi, Gluon_Tran_T0Mu, Gluon_Long_T0Mu)
  # In gap equation, some parts of the kernel do not change during iteration,
  # so we can extract them out.


  kernel(FUNC,ABC) = gap_equation_RL_T0Mu!(T, ABC, FUNC, mu, Mass,
                        N_PV, N_P4, Ap, Bp, Cp,
                        N_QV, N_Q4, Aq, Bq, Cq,
                        AKernelAq, AKernelCq, BKernelBq, CKernelAq, CKernelCq,
                        pv, wpv,qv,wqv,
                        p4, wp4,q4,wq4)
            
  ABC=reinterpret(T,ABC_init) # ABC_init is a complex array, we have to convert it into real array to feed it into non-linear equation solver.
  @time result=nlsolve(kernel, ABC,show_trace=true,method=:anderson,ftol=1e-7)
  ABC=reinterpret(Complex{T},result.zero)
  SplitABC!(N_PV,N_P4,ABC,Ap,Bp,Cp)

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
  pv, qv, p4_real, q4_real,array,
  zphi_array, wphi_array, Gluon_Tran_T0Mu, Gluon_Long_T0Mu) where {T<:AbstractFloat}
  p4 = p4_real + im * Mu
  q4 = q4_real + im * Mu

  for i = 1:N_PHI
    z_phi = zphi_array[i]
    k2_v = pv * pv + qv * qv - 2 * pv * qv * z_phi
    pq = pv * qv * z_phi
    pk = pv * pv - pv * qv * z_phi
    qk = pv * qv * z_phi - qv * qv
    k4 = real(p4 - q4)
    k2 = k2_v + k4 * k4

    GluonZT = Gluon_Tran_T0Mu(k2, Mu)
    GluonZL = Gluon_Long_T0Mu(k2, Mu)


    array[1,i] = (pq * GluonZL + 2.0 * pk * qk / k2 * GluonZL + 2.0 * pk * qk / k2_v * GluonZT - 2.0 * pk * qk / k2_v * GluonZL)
    array[2,i] = GluonZL * (2.0 * q4 * k4 * pk / k2)
    array[3,i] = 2.0 * GluonZT + GluonZL
    array[4,i] = GluonZL * (2.0 * p4 * k4 * qk / k2)
    array[5,i] = p4 * q4 * (2.0 * GluonZT - GluonZL + 2.0 * k4 * k4 / k2 * GluonZL)

  end

  CommonPart = 4.0 / 3.0 / (2 * pi)^3 * qv * qv

  for i=1:N_kernel
    vangle[i] = CommonPart * inte_array_cmplx(view(array,i,:), wphi_array)
  end
end

function Init_Matrix_RL(Mu::T, N_PV, N_QV, N_P4, N_Q4, N_PHI,
  pv_array, qv_array, p4_array, q4_array, zphi_array, wphi_array,
  Gluon_Tran_T0Mu, Gluon_Long_T0Mu) where {T<:AbstractFloat}

  AKernelAq = zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCq = zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelBq = zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAq = zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCq = zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)

  print("Allocating kernel...\n")
  Threads.@threads for n = 1:N_P4
    array = zeros(Complex{T}, N_kernel,N_PHI)
    vangle = zeros(Complex{T}, N_kernel)
    for m = 1:N_Q4, i = 1:N_PV, j = 1:N_QV
      pv = pv_array[i]
      qv = qv_array[j]
      p4 = p4_array[n]
      q4 = q4_array[m]

      Inte_outofz_RL_T0Mu!(Mu, N_PHI, vangle,
        pv, qv, p4, q4,array,
        zphi_array, wphi_array, Gluon_Tran_T0Mu, Gluon_Long_T0Mu)

      AKernelAq[n, m, i, j] = vangle[1]
      AKernelCq[n, m, i, j] = vangle[2]
      BKernelBq[n, m, i, j] = vangle[3]
      CKernelAq[n, m, i, j] = vangle[4]
      CKernelCq[n, m, i, j] = vangle[5]
    end
  end

  return AKernelAq, AKernelCq, BKernelBq, CKernelAq, CKernelCq
end


"""
This is the kernel of the integration equations.
The input variables:
  Ap,Bp,Cp,
  Aq,Bq,Cq,
are matrices for the propagator.
However, you don't need to provide their values.
They are only here to avoid allocating spaces for every iteration.
"""
function gap_equation_RL_T0Mu!(::Type{T}, ABC_real, FUNC_real, Mu, Mass,
  N_PV, N_P4, Ap, Bp, Cp,
  N_QV, N_Q4, Aq_matrix, Bq_matrix, Cq_matrix,
  AKernelAq, AKernelCq, BKernelBq, CKernelAq, CKernelCq,
  pv_array, wpv_array,qv_array,wqv_array,
  p4_array, wp4_array,q4_array,wq4_array) where {T}

  ABC =reinterpret(Complex{T},ABC_real)
  FUNC=reinterpret(Complex{T},FUNC_real)

  
  SplitABC!(N_PV,N_P4,ABC,Ap,Bp,Cp)

  InterpolateABC!(N_PV, N_QV, N_P4, N_Q4,
                         pv_array, qv_array, Ap, Bp, Cp, 
                         Aq_matrix, Bq_matrix, Cq_matrix)



  Threads.@threads for n = 1:N_P4
    for i = 1: N_PV
    	sum_mA = 0.0 +0.0im
    	sum_mB = 0.0 +0.0im
    	sum_mC = 0.0 +0.0im

      p4=p4_array[n]+im*Mu;
    	pv=pv_array[i];

      KerA = zeros(Complex{T}, N_QV)
      KerB = zeros(Complex{T}, N_QV)
      KerC = zeros(Complex{T}, N_QV)
    	for m = 1 : N_Q4
    		for j = 1: N_QV
    			Aq=Aq_matrix[j,m];
    			Bq=Bq_matrix[j,m];
    			Cq=Cq_matrix[j,m];

    			q4=q4_array[m]+im*Mu;
    			qv=qv_array[j];
    			Denorm=1.0/(qv*qv*Aq*Aq+q4*q4*Cq*Cq+Bq*Bq);

    			KerA[j]= Denorm * (AKernelAq[n,m,i,j] * Aq + AKernelCq[n,m,i,j]*Cq );
    			KerB[j]= Denorm * (BKernelBq[n,m,i,j] * Bq);
    			KerC[j]= Denorm * (CKernelAq[n,m,i,j] * Aq + CKernelCq[n,m,i,j]*Cq );

        end
    		sum_mA += wq4_array[m] * inte_array_cmplx(KerA, wqv_array)/(pv*pv);
    		sum_mB += wq4_array[m] * inte_array_cmplx(KerB, wqv_array);
    		sum_mC += wq4_array[m] * inte_array_cmplx(KerC, wqv_array)/(p4*p4);
      end
    	FUNC[(n-1) * 3 * N_PV + i]            = Ap[i,n]- 1    - sum_mA ;
    	FUNC[(n-1) * 3 * N_PV + N_PV + i]     = Bp[i,n]- Mass - sum_mB ;
    	FUNC[(n-1) * 3 * N_PV + 2 * N_PV + i] = Cp[i,n]- 1    - sum_mC;
    end
  end
end

end #end module
