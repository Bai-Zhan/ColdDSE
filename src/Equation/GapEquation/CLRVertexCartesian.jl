module CLRVertexCartesian

using Printf
using GaussLegendrePolynomial
using NLsolve, Dierckx
using HDF5
using GluonModel
using Revise

const N_kernel = 35::Integer

function inte_array_cmplx(func, w) 
  s = zero(eltype(func))
  for i = eachindex(func)
    s += func[i] * w[i]
  end
  return s
end

include(joinpath(@__DIR__,"CommonFunction","AllocateCommonVariables.jl"))
include(joinpath(@__DIR__,"CommonFunction","AssignInitValue.jl"))
include(joinpath(@__DIR__,"CommonFunction","TransformABC.jl"))
include(joinpath(@__DIR__,"CommonFunction","InterpolateABC.jl"))
include(joinpath(@__DIR__,"..","..","Constants.jl"))

function solve(::Type{T},constants;result_file::String) where{T<:Real}
  solve(T, constants.mass, constants.mu; 
                constants.init_type,constants.gluon_model,constants.eta,constants.pole_handle_s,
                constants.Gluon_D_Long, constants.Gluon_D_Tran, constants.Gluon_omega, constants.LambdaR_P4, constants.LambdaV_P4, 
                constants.LambdaR_PV, constants.LambdaV_PV, 
                constants.N_PV, constants.N_QV, constants.N_P4, constants.N_PHI,
                result_file)
end

function solve(::Type{T}, Mass::T, mu::T; 
                init_type,gluon_model,eta,pole_handle_s,
                Gluon_D_Long::T, Gluon_D_Tran::T, Gluon_omega::T, LambdaR_P4::T, LambdaV_P4::T, 
                LambdaR_PV::T, LambdaV_PV::T, 
                N_PV, N_QV, N_P4, N_PHI,
                result_file::String) where {T<:AbstractFloat}
  if N_PV==N_QV
    print("Warning. N_PV==N_QV. Might lead to singularity.")
  end

  N_Q4,
  pv,wpv,qv,wqv,
  p4,wp4,q4,wq4,
  zphi,wphi,
  Ap,Bp,Cp,Aq,Bq,Cq,ABC_init =AllocateCommonVariables_Cartesian(N_PV,N_QV,N_P4,N_PHI,LambdaR_P4,LambdaV_P4,LambdaR_PV,LambdaV_PV)

  AssignInitValue!(init_type, Mass, N_PV, N_P4, Ap, Bp, Cp,ABC_init)

  Gluon_Long_T0Mu, Gluon_Tran_T0Mu = GluonModel.InitGluonModel(gluon_model,Gluon_D_Long,Gluon_D_Tran,Gluon_omega)

  @time AKernelApAq, AKernelApCq, AKernelCpAq, AKernelCpCq, 
  AKernelAqAq, AKernelAqCq, AKernelAqDA, AKernelAqDC, AKernelAqt4, 
  AKernelBqDB, AKernelCqCq, AKernelCqDA, AKernelCqDC, AKernelCqt4, 
  BKernelApBq, BKernelCpBq, BKernelAqBq, BKernelAqDB, 
  BKernelBqCq, BKernelBqDA, BKernelBqDC, BKernelCqDB, 
  CKernelApAq, CKernelApCq, CKernelCpAq, CKernelCpCq, 
  CKernelAqAq, CKernelAqCq, CKernelAqDA, CKernelAqDC, CKernelAqt4,
  CKernelBqDB, CKernelCqCq, CKernelCqDA, CKernelCqDC=Init_Matrix(mu, N_PV, N_QV, N_P4, N_Q4, N_PHI,
                                                      pv, qv, p4, q4, zphi, wphi,
                                                      Gluon_Tran_T0Mu, Gluon_Long_T0Mu,eta)
  # In gap equation, some parts of the kernel do not change during iteration,
  # so we can extract them out.


kernel(FUNC,ABC)=gap_equation!(T, ABC, FUNC, mu, Mass,
  N_PV, N_P4, Ap, Bp, Cp,
  N_QV, N_Q4, Aq, Bq, Cq,
  AKernelApAq, AKernelApCq, AKernelCpAq, AKernelCpCq, AKernelAqAq, AKernelAqCq, AKernelAqDA, AKernelAqDC, AKernelAqt4, 
  AKernelBqDB, AKernelCqCq, AKernelCqDA, AKernelCqDC, AKernelCqt4, BKernelApBq, BKernelCpBq, BKernelAqBq, BKernelAqDB, 
  BKernelBqCq, BKernelBqDA, BKernelBqDC, BKernelCqDB, CKernelApAq, CKernelApCq,
  CKernelCpAq, CKernelCpCq, CKernelAqAq, CKernelAqCq, CKernelAqDA, CKernelAqDC, CKernelAqt4,
  CKernelBqDB, CKernelCqCq, CKernelCqDA, CKernelCqDC,
  pv, wpv,qv,wqv,
  p4, wp4,q4,wq4,eta,pole_handle_s)

            
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

function Inte_outofz(Mu::T, N_PHI::Integer, vangle,
  pv, qv, p4_real, q4_real,array,
  zphi_array, wphi_array, Gluon_Tran_T0Mu, Gluon_Long_T0Mu,eta) where {T<:AbstractFloat}
  p4 = p4_real + im * Mu
  q4 = q4_real + im * Mu

  for i = 1:N_PHI
    z_phi = zphi_array[i]
    k2_v = pv*pv + qv*qv - 2 * pv*qv*z_phi;

    pq_v = pv*qv*z_phi;
    pk_v = pv*pv - pv*qv*z_phi;
    kq_v = pv*qv*z_phi - qv*qv;

    k4=real(p4-q4);
    k2 = k2_v + k4*k4;
    p2=pv*pv+p4*p4;
    q2=qv*qv+q4*q4;

    pq=pq_v+p4*q4;
    pk=pk_v+p4*k4;
    kq=kq_v+k4*q4;


    DT = Gluon_Tran_T0Mu(k2, Mu)
    DL = Gluon_Long_T0Mu(k2, Mu)

    array[ 1,i]=(DT*k2*(4*k4*kq*p4 - 4*kq*pk - 4*(k4*k4)*p4*q4 + 4*k4*pk*q4) + DL*(k4*k4)*(4*kq*pk - 2*k2*pq + 2*k2*p4*q4 + (k4*k4)*(2*pq + 2*p4*q4) + k4*(-4*kq*p4 - 4*pk*q4)))/(k2*(k2 - 1*(k4*k4)));
    array[ 2,i]=(DL*k4*(2*k2*k4*p4 - 2*(k4*k4*k4)*p4 - 2*k2*pk + 2*(k4*k4)*pk)*q4)/(k2*(k2 - 1*(k4*k4)));
    array[ 3,i]=(DL*(k2*(k4*k4)*(4*pq - 4*p4*q4) + (k2*k2)*(-2*pq + 2*p4*q4) + (k4*k4*k4*k4)*(-2*pq + 2*p4*q4)))/(k2*(k2 - 1*(k4*k4)));
    array[ 4,i]=(DL*k4*(2*k2*k4*p4 - 2*(k4*k4*k4)*p4 - 2*k2*pk + 2*(k4*k4)*pk)*q4)/(k2*(k2 - 1*(k4*k4)));
    array[ 5,i]=(DT*k2*(4*k4*kq*p4 - 4*kq*pk - 4*(k4*k4)*p4*q4 + 4*k4*pk*q4) + DL*(k4*k4)*(4*kq*pk - 2*k2*pq + 2*k2*p4*q4 + (k4*k4)*(2*pq + 2*p4*q4) + k4*(-4*kq*p4 - 4*pk*q4)))/(k2*(k2 - 1*(k4*k4)));
    array[ 6,i]=(DL*((k2*k2)*(-2*pq + 2*p4*q4) + k2*k4*(4*k4*pq - 2*k4*p4*q4 - 2*pk*q4) + (k4*k4*k4)*(-2*k4*pq + 2*pk*q4)))/(k2*(k2 - 1*(k4*k4)));
    array[ 7,i]= DT*(2*pq*q2 + 2*(p4*p4*p4)*q4 - 2*pq*(q4*q4) + p2*(2*pq + 4*q2 - 2*p4*q4 - 4*(q4*q4)) + (p4*p4)*(-2*pq - 4*q2 + 4*(q4*q4)) + p4*(-2*q2*q4 + 2*(q4*q4*q4))) + (complex(2.,0.)*(DT*k2 - 1*DL*(k4*k4))*((kq*kq)*(-1*p2 + 1*(p4*p4)) + kq*(-1*p2*pk + 1*(p4*p4)*pk + 1*k4*p4*q2 - 1*pk*q2 + 1*k4*p2*q4 - 1*k4*(p4*p4)*q4 - 1*k4*p4*(q4*q4) + 1*pk*(q4*q4)) + pk*(-1*pk*q2 + 1*k4*p2*q4 + 1*pk*(q4*q4) + k4*p4*(1*q2 + (-1*p4 - 1*q4)*q4))))/(k2*(k2 - 1*(k4*k4))) - (complex(2.,0.)*(DL - 1*DT)*k4*(kq*(-1*p2*p4 + 1*(p4*p4*p4) - 1*p2*q4 + 1*(p4*p4)*q4) + pk*(-1*p4*q2 - 1*q2*q4 + 1*p4*(q4*q4) + 1*(q4*q4*q4)) + k4*(-1*(p4*p4*p4)*q4 + 1*p2*(q4*q4) + (p4*p4)*(1*q2 - 2*(q4*q4)) + p4*q4*(1*p2 + 1*q2 - 1*(q4*q4)))))/(-1*k2 + (k4*k4));
    array[ 8,i]= (DL*((k4*k4*k4)*(kq + pk)*(-2. *p4*pq + 2. *(p4*p4)*q4 - 2. *pq*q4 + 2. *p4*(q4*q4)) + k2*k4*(p4*(2. *kq + 2. *k4*p4 + 2. *pk)*pq + (-2. *kq*(p4*p4) - 2. *k4*(p4*p4*p4) - 2. *(p4*p4)*pk + 2. *kq*pq + 4. *k4*p4*pq + 2. *pk*pq)*q4 + (-2. *kq*p4 - 4. *k4*(p4*p4) - 2. *p4*pk + 2. *k4*pq)*(q4*q4) - 2. *k4*p4*(q4*q4*q4)) + (k2*k2)*(2. *(p4*p4*p4)*q4 - 2. *pq*(q4*q4) + (p4*p4)*(-2. *pq + 4. *(q4*q4)) + p4*(-4. *pq*q4 + 2. *(q4*q4*q4)))))/(k2*(k2 - 1. *(k4*k4)));
    array[ 9,i]= (complex(-2.,0.)*(DL + DT)*((k4*k4)*(kq*(-1. *p2 + 1. *(p4*p4) - 1. *pq + 1. *p4*q4) + k4*(-1. *p4*pq - 1. *p4*q2 + 1. *p2*q4 + 1. *pq*q4) + pk*(1. *pq + 1. *q2 - 1. *p4*q4 - 1. *(q4*q4))) + k2*(kq*(1. *p2 - 1. *(p4*p4) + 1. *pq - 1. *p4*q4) + k4*(1. *p4*pq + 1. *p4*q2 - 1. *p2*q4 - 1. *pq*q4) + pk*(-1. *pq - 1. *q2 + 1. *p4*q4 + 1. *(q4*q4)))))/(1. *k2 - 1. *(k4*k4));
    array[10,i]= (complex(0.,0.) + DT*k2*(-8. *eta*(k4*k4*k4)*p4 + (-4. *kq - 4. *pk)*pk + (k4*k4)*(-4. *p2 + 8. *eta*pk - 4. *pq) + k2*(4. *p2 + 8. *eta*k4*p4 - 4. *(p4*p4) - 8. *eta*pk + 4. *pq - 4. *p4*q4) + k4*(4. *kq*p4 + 8. *p4*pk + 4. *pk*q4)) + DL*(eta*k2*(4. *k2*k4*p4 - 4. *(k4*k4*k4)*p4 - 4. *k2*pk + 4. *(k4*k4)*pk) + k4*((k4*k4)*p4*(-4. *kq - 4. *pk) + k4*pk*(4. *kq + 4. *pk) + k2*pk*(-4. *p4 - 4. *q4) + k2*k4*p4*(4. *p4 + 4. *q4))))/(k2*(k2 - 1. *(k4*k4)));
    array[11,i]=(DL*k4*(2. *k2*k4*p4 - 2. *(k4*k4*k4)*p4 - 2. *k2*pk + 2. *(k4*k4)*pk)*q4)/(k2*(k2 - 1. *(k4*k4)));
    array[12,i]= (DL*q4*(k2*k4*(kq + k4*p4 + pk)*(-2. *p2 + 2. *(p4*p4) - 2. *pq) + (k2*k2)*p4*(2. *p2 - 2. *(p4*p4) + 2. *pq) + (k4*k4*k4)*(kq + pk)*(2. *p2 - 2. *(p4*p4) + 2. *pq) + ((k4*k4*k4)*p4*(-2. *kq - 2. *pk) + k2*k4*(2. *kq*p4 + 2. *p4*pk + k4*(-2. *p2 + 4. *(p4*p4) - 2. *pq)) + (k2*k2)*(2. *p2 - 4. *(p4*p4) + 2. *pq))*q4 + k2*(-2. *k2 + 2. *(k4*k4))*p4*(q4*q4)))/(k2*(k2 - 1. *(k4*k4)));
    array[13,i]= (q4*(DL*k4*((k4*k4)*p4*(-2. *kq*p4 - 2. *p4*pk - 2. *kq*q4 - 2. *pk*q4) + k4*pk*(2. *kq*p4 + 2. *p4*pk + 2. *kq*q4 + 2. *pk*q4) + k2*pk*(-2. *(p4*p4) - 4. *p4*q4 - 2. *(q4*q4)) + k2*k4*p4*(2. *(p4*p4) + 4. *p4*q4 + 2. *(q4*q4))) + DT*k2*(k4*kq*p4*(2. *p4 + 2. *q4) + pk*(-2. *kq*p4 - 2. *p4*pk - 2. *kq*q4 - 2. *pk*q4) + (k4*k4)*(-2. *p2*p4 - 2. *p4*pq - 2. *p2*q4 - 2. *pq*q4) + k4*pk*(4. *(p4*p4) + 6. *p4*q4 + 2. *(q4*q4)) + k2*(2. *p2*p4 - 2. *(p4*p4*p4) + 2. *p4*pq + 2. *p2*q4 - 4. *(p4*p4)*q4 + 2. *pq*q4 - 2. *p4*(q4*q4)))))/(k2*(1. *k2 - 1. *(k4*k4)));
    array[14,i]= (q4*(DL*k4*((-2. *kq - 2. *pk)*pk + (k4*k4)*(-2. *p2 - 2. *pq) + k2*(2. *p2 - 2. *(p4*p4) + 2. *pq - 2. *p4*q4) + k4*(2. *kq*p4 + 4. *p4*pk + 2. *pk*q4)) + DT*(k2*(-4. *p4*pk - 4. *pk*q4 + k4*(2. *p2 + 2. *(p4*p4) + 2. *pq + 2. *p4*q4)) + k4*(pk*(2. *kq + 2. *pk) + (k4*k4)*(-2. *p2 - 2. *pq) + k4*(-2. *kq*p4 + 2. *pk*q4)))))/(-1. *k2 + (k4*k4));
    array[15,i]=4. *DT + (2. *DL*(k4*k4))/k2;
    array[16,i]=DL*(2. - (2. *(k4*k4))/k2);
    array[17,i]=4. *DT + (2. *DL*(k4*k4))/k2;
    array[18,i]= (2. *(-2. *(DL - 1. *DT)*eta*(k2*k2)*(1. *kq - 1. *k4*q4) + (DT*k2 - DL*(k4*k4))*(2. *(kq*kq) + 2. *kq*pk - 2. *k4*kq*q4 - 2. *k4*pk*q4) + 6. *DT*k2*(k2 - 1. *(k4*k4))*(-1. *eta*kq - 0.3333333333333333*pq - 0.3333333333333333*q2 + 1. *eta*k4*q4 + 0.3333333333333333*p4*q4 + 0.3333333333333333*(q4*q4)) + (DL - DT)*k2*k4*(k4*(-2. *p4 - 2. *q4)*q4 + kq*(2. *p4 + 2. *q4) + eta*k4*(2. *kq - 2. *k4*q4))))/(k2*(k2 - (k4*k4)));
    array[19,i]=DL*(2. - (2. *(k4*k4))/k2);
    array[20,i]= 2. *(DT*(1. *p2 - 1. *(p4*p4) + 2. *pq + 1. *q2 - 2. *p4*q4 - 1. *(q4*q4)) + ((DL - DT)*k4*(1. *kq*p4 - 1. *k4*(p4*p4) + 1. *p4*pk + 1. *kq*q4 - 2. *k4*p4*q4 + 1. *pk*q4 - 1. *k4*(q4*q4)))/(-k2 + (k4*k4)) + ((-(DT*k2) + DL*(k4*k4))*(1. *(kq*kq) + pk*(-1. *k4*p4 + 1. *pk - 1. *k4*q4) + kq*(-1. *k4*p4 + 2. *pk - 1. *k4*q4)))/(k2*(k2 - (k4*k4))));
    array[21,i]=(DL*((k4*k4*k4)*(2. *kq*p4 + 2. *p4*pk + 2. *kq*q4 + 2. *pk*q4) + (k2*k2)*(2. *(p4*p4) + 4. *p4*q4 + 2. *(q4*q4)) + k2*k4*(-2. *kq*p4 - 2. *k4*(p4*p4) - 2. *p4*pk - 2. *kq*q4 - 4. *k4*p4*q4 - 2. *pk*q4 - 2. *k4*(q4*q4))))/(k2*(k2 - 1. *(k4*k4)));
    array[22,i]=(q4*(DT*eta*k2*k4*(-8. *k2 + 8. *(k4*k4)) + DL*(eta*k2*k4*(-4. *k2 + 4. *(k4*k4)) + (k4*k4*k4)*(-4. *kq - 4. *pk) + (k2*k2)*(-4. *p4 - 4. *q4) + k2*k4*(4. *kq + 4. *k4*p4 + 4. *pk + 4. *k4*q4))))/(k2*(k2 - 1. *(k4*k4)));
    array[23,i]=(DL*k4*p4*(-2. *k2*kq + 2. *(k4*k4)*kq + 2. *k2*k4*q4 - 2. *(k4*k4*k4)*q4))/(k2*(k2 - 1. *(k4*k4)));
    array[24,i]=((DT*k2*(-4. *k2 + 4. *(k4*k4)) + DL*(-2. *k2*(k4*k4) + 2. *(k4*k4*k4*k4)))*p4*q4)/(k2*(k2 - 1. *(k4*k4)));
    array[25,i]=(DL*k4*p4*(-2. *k2*kq + 2. *(k4*k4)*kq + 2. *k2*k4*q4 - 2. *(k4*k4*k4)*q4))/(k2*(k2 - 1. *(k4*k4)));
    array[26,i]=(DL*(2. *(k2*k2) - 4. *k2*(k4*k4) + 2. *(k4*k4*k4*k4))*p4*q4)/(k2*(k2 - 1. *(k4*k4)));
    array[27,i]=(DL*k4*p4*(-2. *k2*kq + 2. *(k4*k4)*kq + 2. *k2*k4*q4 - 2. *(k4*k4*k4)*q4))/(k2*(k2 - 1. *(k4*k4)));
    array[28,i]=(p4*(DL*k4*(-2. *k2 + 2. *(k4*k4))*kq + DT*k2*(-4. *k2 + 4. *(k4*k4))*q4))/(k2*(k2 - 1. *(k4*k4)));
    array[29,i]= (DL*p4*((k4*k4*k4)*(kq + pk)*(2. *pq + 2. *q2 + (-2. *p4 - 2. *q4)*q4) + (k2*k2)*(-2. *(p4*p4)*q4 + 2. *pq*q4 + 2. *q2*q4 - 2. *(q4*q4*q4) + p4*(2. *pq + 2. *q2 - 4. *(q4*q4))) + k2*k4*(kq*(-2. *pq - 2. *q2 + 2. *p4*q4 + 2. *(q4*q4)) + pk*(-2. *pq - 2. *q2 + 2. *p4*q4 + 2. *(q4*q4)) + k4*(-2. *p4*pq - 2. *p4*q2 + 2. *(p4*p4)*q4 - 2. *pq*q4 - 2. *q2*q4 + 4. *p4*(q4*q4) + 2. *(q4*q4*q4)))))/(k2*(k2 - 1. *(k4*k4)));
    array[30,i]= (p4*(DL*k4*((k4*k4)*q4*(-2. *kq*p4 - 2. *p4*pk - 2. *kq*q4 - 2. *pk*q4) + k2*kq*(-2. *(p4*p4) - 4. *p4*q4 - 2. *(q4*q4)) + k4*((kq*kq)*(2. *p4 + 2. *q4) + kq*pk*(2. *p4 + 2. *q4) + k2*q4*(2. *(p4*p4) + 4. *p4*q4 + 2. *(q4*q4)))) + DT*k2*((kq*kq)*(-2. *p4 - 2. *q4) + k4*(pk*q4*(2. *p4 + 2. *q4) + k4*(-2. *p4*pq - 2. *p4*q2 - 2. *pq*q4 - 2. *q2*q4)) + k2*(-2. *(p4*p4)*q4 + 2. *pq*q4 + 2. *q2*q4 - 2. *(q4*q4*q4) + p4*(2. *pq + 2. *q2 - 4. *(q4*q4))) + kq*(-2. *p4*pk - 2. *pk*q4 + k4*(2. *(p4*p4) + 6. *p4*q4 + 4. *(q4*q4))))))/(k2*(k2 - 1. *(k4*k4)));
    array[31,i]= complex(-4.,0.)*p4*(DT*(1. *kq*p4 - 1. *k4*pq - 1. *k4*q2 + 1. *kq*q4) - (complex(0.5,0.)*(DL - 1. *DT)*k4*(1. *(kq*kq) + kq*(-1. *k4*p4 + 1. *pk - 2. *k4*q4) + k4*(1. *k4*pq + 1. *k4*q2 - 1. *pk*q4) + k2*(-1. *pq - 1. *q2 + 1. *p4*q4 + 1. *(q4*q4))))/ (-1. *k2 + (k4*k4)));
    array[32,i]=(p4*(DT*eta*k2*k4*(-8. *k2 + 8. *(k4*k4)) + DL*(eta*k2*k4*(-4. *k2 + 4. *(k4*k4)) + (k4*k4*k4)*(4. *kq + 4. *pk) + (k2*k2)*(4. *p4 + 4. *q4) + k2*k4*(-4. *kq - 4. *k4*p4 - 4. *pk - 4. *k4*q4))))/(k2*(k2 - 1. *(k4*k4)));
    array[33,i]=(DL*(2. *(k2*k2) - 4. *k2*(k4*k4) + 2. *(k4*k4*k4*k4))*p4*q4)/(k2*(k2 - 1. *(k4*k4)));
    array[34,i]= (p4*q4*(DT*k2*(-2. *(kq*kq) - 2. *(k4*k4)*p2 + 4. *k4*p4*pk - 2. *(pk*pk) - 4. *(k4*k4)*pq - 2. *(k4*k4)*q2 + 4. *k4*pk*q4 + kq*(4. *k4*p4 - 4. *pk + 4. *k4*q4) + k2*(2. *p2 - 2. *(p4*p4) + 4. *pq + 2. *q2 - 4. *p4*q4 - 2. *(q4*q4))) + DL*k4*(k2*(-2. *kq*p4 - 2. *p4*pk - 2. *kq*q4 - 2. *pk*q4) + (k4*k4)*(-2. *kq*p4 - 2. *p4*pk - 2. *kq*q4 - 2. *pk*q4) + k4*(2. *(kq*kq) + 2. *k2*(p4*p4) + 4. *kq*pk + 2. *(pk*pk) + 4. *k2*p4*q4 + 2. *k2*(q4*q4)))))/ (k2*(-1. *k2 + 1. *(k4*k4)));
    array[35,i]= (DL*p4*q4*((k4*k4*k4)*(2. *kq*p4 + 2. *p4*pk + 2. *kq*q4 + 2. *pk*q4) + (k2*k2)*(2. *(p4*p4) + 4. *p4*q4 + 2. *(q4*q4)) + k2*k4*(-2. *kq*p4 - 2. *k4*(p4*p4) - 2. *p4*pk - 2. *kq*q4 - 4. *k4*p4*q4 - 2. *pk*q4 - 2. *k4*(q4*q4))))/ (k2*(k2 - 1. *(k4*k4)));
  end

  CommonPart = 4.0 / 3.0 / (2 * pi)^3 * qv * qv

  for i=1:N_kernel
    vangle[i] = CommonPart * inte_array_cmplx(view(array,i,:), wphi_array)
  end
end

function Init_Matrix(Mu::T, N_PV, N_QV, N_P4, N_Q4, N_PHI,
  pv_array, qv_array, p4_array, q4_array, zphi_array, wphi_array,
  Gluon_Tran_T0Mu, Gluon_Long_T0Mu,eta) where {T<:AbstractFloat}


  AKernelApAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelApCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCpAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCpCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelAqAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelAqCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelAqDA= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelAqDC= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelAqt4= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelBqDB= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCqCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCqDA= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCqDC= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  AKernelCqt4= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  
  BKernelApBq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelCpBq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelAqBq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelAqDB= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelBqCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelBqDA= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelBqDC= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  BKernelCqDB= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)

  CKernelApAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelApCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCpAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCpCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAqAq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAqCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAqDA= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAqDC= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelAqt4= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelBqDB= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCqCq= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCqDA= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)
  CKernelCqDC= zeros(Complex{T}, N_P4, N_Q4, N_PV, N_QV)

  print("Allocating kernel...\n")
  Threads.@threads for n = 1:N_P4

    array = zeros(Complex{T}, N_kernel,N_PHI)
    vangle = zeros(Complex{T}, N_kernel)
    for m = 1:N_Q4, i = 1:N_PV, j = 1:N_QV
      pv = pv_array[i]
      qv = qv_array[j]
      p4 = p4_array[n]
      q4 = q4_array[m]

      Inte_outofz(Mu, N_PHI, vangle,
        pv, qv, p4, q4,array,
        zphi_array, wphi_array, Gluon_Tran_T0Mu, Gluon_Long_T0Mu,eta)

        AKernelApAq[n,m,i,j]=vangle[1]
        AKernelApCq[n,m,i,j]=vangle[2]
        AKernelCpAq[n,m,i,j]=vangle[3]
        AKernelCpCq[n,m,i,j]=vangle[4]
        AKernelAqAq[n,m,i,j]=vangle[5]
        AKernelAqCq[n,m,i,j]=vangle[6]
        AKernelAqDA[n,m,i,j]=vangle[7]
        AKernelAqDC[n,m,i,j]=vangle[8]
        AKernelAqt4[n,m,i,j]=vangle[9]
        AKernelBqDB[n,m,i,j]=vangle[10]
        AKernelCqCq[n,m,i,j]=vangle[11]
        AKernelCqDA[n,m,i,j]=vangle[12]
        AKernelCqDC[n,m,i,j]=vangle[13]
        AKernelCqt4[n,m,i,j]=vangle[14]

        BKernelApBq[n,m,i,j]=vangle[15]
        BKernelCpBq[n,m,i,j]=vangle[16]
        BKernelAqBq[n,m,i,j]=vangle[17]
        BKernelAqDB[n,m,i,j]=vangle[18]
        BKernelBqCq[n,m,i,j]=vangle[19]
        BKernelBqDA[n,m,i,j]=vangle[20]
        BKernelBqDC[n,m,i,j]=vangle[21]
        BKernelCqDB[n,m,i,j]=vangle[22]

        CKernelApAq[n,m,i,j]=vangle[23]
        CKernelApCq[n,m,i,j]=vangle[24]
        CKernelCpAq[n,m,i,j]=vangle[25]
        CKernelCpCq[n,m,i,j]=vangle[26]
        CKernelAqAq[n,m,i,j]=vangle[27]
        CKernelAqCq[n,m,i,j]=vangle[28]
        CKernelAqDA[n,m,i,j]=vangle[29]
        CKernelAqDC[n,m,i,j]=vangle[30]
        CKernelAqt4[n,m,i,j]=vangle[31]
        CKernelBqDB[n,m,i,j]=vangle[32]
        CKernelCqCq[n,m,i,j]=vangle[33]
        CKernelCqDA[n,m,i,j]=vangle[34]
        CKernelCqDC[n,m,i,j]=vangle[35];

    end
  end

  return AKernelApAq, AKernelApCq, AKernelCpAq, AKernelCpCq, AKernelAqAq, AKernelAqCq, AKernelAqDA, AKernelAqDC, AKernelAqt4, 
    AKernelBqDB, AKernelCqCq, AKernelCqDA, AKernelCqDC, AKernelCqt4, BKernelApBq, BKernelCpBq, BKernelAqBq, BKernelAqDB, 
    BKernelBqCq, BKernelBqDA, BKernelBqDC, BKernelCqDB, CKernelApAq, CKernelApCq,
    CKernelCpAq, CKernelCpCq, CKernelAqAq, CKernelAqCq, CKernelAqDA, CKernelAqDC, CKernelAqt4,
    CKernelBqDB, CKernelCqCq, CKernelCqDA, CKernelCqDC
end


function DeltaF(Fp,Fq,dFdpv,dFdp4,pv,qv,p4,q4,mu,pole_handle_s)
  s=pole_handle_s; # a smaller s means you do less substraction, but will be difficult for the Dyson-Schwinger equation to converge.
  diff =(Fp-Fq)/(pv*pv-qv*qv+(p4+im*mu)*(p4+im*mu)-(q4+im*mu)*(q4+im*mu));
  temp =abs2(pv*pv-qv*qv+p4*p4-q4*q4+2*im*mu*(p4-q4))/s;
  weight = 1 - exp(-temp);


  #deriv=dFdpv*dFdp4/(2*pv*dFdp4+2*(p4+im*mu)*dFdpv);
  #if !((abs(pv*pv - qv*qv + p4*p4 - q4*q4)<1e-40) && (abs(mu*(p4 - q4))<1e-40))
  #  Delta=diff*weight+deriv*(1-weight);
  #else  Delta=deriv;
  #end

  Delta=diff*weight
  if abs(Delta) < 1e-10
    Delta=0
  end


  return Delta;
end

function dABC_dp!(N_PV,N_P4,pv_array,p4_array,F,dFdpv,dFdp4)
  T=eltype(pv_array)

  tempFreal_pv=zeros(T,N_PV)
  tempFimag_pv=zeros(T,N_PV)
  for n=1:N_P4
    for i=1:N_PV
      tempFreal_pv[i]=real(F[i,n]);
      tempFimag_pv[i]=imag(F[i,n]);
    end
    realFspline = Spline1D(pv_array, tempFreal_pv)
    imagFspline = Spline1D(pv_array, tempFimag_pv)

    for i=1:N_PV
      dFdpv[i,n]=complex(derivative(realFspline,pv_array[i]),derivative(imagFspline,pv_array[i]))
    end
  end

  tempFreal_p4=zeros(T,N_P4)
  tempFimag_p4=zeros(T,N_P4)
  for i=1:N_PV
    for n=1:N_P4
      tempFreal_p4[n]=real(F[i,n]);
      tempFimag_p4[n]=imag(F[i,n]);
    end
    realFspline = Spline1D(p4_array, tempFreal_p4)
    imagFspline = Spline1D(p4_array, tempFimag_p4)

    for n=1:N_P4
      dFdp4[i,n]=complex(derivative(realFspline,p4_array[n]),derivative(imagFspline,p4_array[n]))
    end
  end
end

function dABC_dp(N_PV,N_P4,pv_array,p4_array,A,B,C)
  dAdpv=zeros(complex(eltype(pv_array)),N_PV,N_P4)
  dAdp4=zeros(complex(eltype(pv_array)),N_PV,N_P4)

  dBdpv=zeros(complex(eltype(pv_array)),N_PV,N_P4)
  dBdp4=zeros(complex(eltype(pv_array)),N_PV,N_P4)

  dCdpv=zeros(complex(eltype(pv_array)),N_PV,N_P4)
  dCdp4=zeros(complex(eltype(pv_array)),N_PV,N_P4)

  dABC_dp!(N_PV,N_P4,pv_array,p4_array,A,dAdpv,dAdp4)
  dABC_dp!(N_PV,N_P4,pv_array,p4_array,B,dBdpv,dBdp4)
  dABC_dp!(N_PV,N_P4,pv_array,p4_array,C,dCdpv,dCdp4)
  return dAdpv, dAdp4, dBdpv, dBdp4, dCdpv, dCdp4
end


function gap_equation!(::Type{T}, ABC_real, FUNC_real, Mu, Mass,
  N_PV, N_P4, Ap_matrix, Bp_matrix, Cp_matrix,
  N_QV, N_Q4, Aq_matrix, Bq_matrix, Cq_matrix,
  AKernelApAq, AKernelApCq, AKernelCpAq, AKernelCpCq, AKernelAqAq, AKernelAqCq, AKernelAqDA, AKernelAqDC, AKernelAqt4, 
  AKernelBqDB, AKernelCqCq, AKernelCqDA, AKernelCqDC, AKernelCqt4, BKernelApBq, BKernelCpBq, BKernelAqBq, BKernelAqDB, 
  BKernelBqCq, BKernelBqDA, BKernelBqDC, BKernelCqDB, CKernelApAq, CKernelApCq,
  CKernelCpAq, CKernelCpCq, CKernelAqAq, CKernelAqCq, CKernelAqDA, CKernelAqDC, CKernelAqt4,
  CKernelBqDB, CKernelCqCq, CKernelCqDA, CKernelCqDC,
  pv_array, wpv_array,qv_array,wqv_array,
  p4_array, wp4_array,q4_array,wq4_array,eta,pole_handle_s) where {T}

  ABC =reinterpret(Complex{T},ABC_real)
  FUNC=reinterpret(Complex{T},FUNC_real)

  
  SplitABC!(N_PV,N_P4,ABC,Ap_matrix,Bp_matrix,Cp_matrix)
  InterpolateABC!(N_PV, N_QV, N_P4, N_Q4, pv_array, qv_array, Ap_matrix, Bp_matrix, Cp_matrix, Aq_matrix, Bq_matrix, Cq_matrix)
  dAdpv, dAdp4, dBdpv, dBdp4, dCdpv, dCdp4 = dABC_dp(N_PV,N_P4,pv_array,p4_array,Ap_matrix,Bp_matrix,Cp_matrix)

  Threads.@threads for n = 1:N_P4
    for i = 1: N_PV
      sum_mA = 0.0 +0.0im
      sum_mB = 0.0 +0.0im
      sum_mC = 0.0 +0.0im

      p4=p4_array[n]+im*Mu;
      pv=pv_array[i];			
      p2=pv*pv+p4*p4;


      KerA = zeros(Complex{T}, N_QV)
      KerB = zeros(Complex{T}, N_QV)
      KerC = zeros(Complex{T}, N_QV)
      for m = 1 : N_Q4
        for j = 1: N_QV
          Ap=Ap_matrix[i,n];
          Bp=Bp_matrix[i,n];
          Cp=Cp_matrix[i,n];

          Aq=Aq_matrix[j,m];
          Bq=Bq_matrix[j,m];
          Cq=Cq_matrix[j,m];

          q4=q4_array[m]+im*Mu;
          qv=qv_array[j];
          q2=qv*qv+q4*q4;

          Denorm=1.0/(qv*qv*Aq*Aq+q4*q4*Cq*Cq+Bq*Bq);

          DA=DeltaF(Ap,Aq,dAdpv[i,n],dAdp4[i,n],pv,qv,real(p4),real(q4),Mu,pole_handle_s);
          DB=DeltaF(Bp,Bq,dBdpv[i,n],dBdp4[i,n],pv,qv,real(p4),real(q4),Mu,pole_handle_s);
          DC=DeltaF(Cp,Cq,dCdpv[i,n],dCdp4[i,n],pv,qv,real(p4),real(q4),Mu,pole_handle_s);
          Mp=Bp/Ap;
          Mq=Bq/Aq;
          t5=eta*DB;
          t4=2*t5*2*(Mp+Mq)/(p2+q2+Mp*Mp+Mq*Mq);
          KerA[j]=Denorm*( AKernelApAq[n,m,i,j] * Ap * Aq   +  AKernelApCq[n,m,i,j] * Ap * Cq   +  AKernelCpAq[n,m,i,j] * Cp * Aq  +  AKernelCpCq[n,m,i,j] * Cp * Cq   
                        +  AKernelAqAq[n,m,i,j] * Aq * Aq   +  AKernelAqCq[n,m,i,j] * Aq * Cq   +  AKernelAqDA[n,m,i,j] * Aq * DA  +  AKernelAqDC[n,m,i,j] * Aq * DC   
                        +  AKernelAqt4[n,m,i,j] * Aq * t4   +  AKernelBqDB[n,m,i,j] * Bq * DB   +  AKernelCqCq[n,m,i,j] * Cq * Cq  +  AKernelCqDA[n,m,i,j] * Cq * DA   
                        +  AKernelCqDC[n,m,i,j] * Cq * DC   +  AKernelCqt4[n,m,i,j] * Cq * t4)

          KerB[j]=Denorm*( BKernelApBq[n,m,i,j] * Ap * Bq   +  BKernelCpBq[n,m,i,j] * Cp * Bq   +  BKernelAqBq[n,m,i,j] * Aq * Bq  +  BKernelAqDB[n,m,i,j] * Aq * DB
                        +  BKernelBqCq[n,m,i,j] * Bq * Cq   +  BKernelBqDA[n,m,i,j] * Bq * DA   +  BKernelBqDC[n,m,i,j] * Bq * DC  +  BKernelCqDB[n,m,i,j] * Cq * DB);

          KerC[j]=Denorm*( CKernelApAq[n,m,i,j] * Ap * Aq   +  CKernelApCq[n,m,i,j] * Ap * Cq   +  CKernelCpAq[n,m,i,j] * Cp * Aq  +  CKernelCpCq[n,m,i,j] * Cp * Cq   
                        +  CKernelAqAq[n,m,i,j] * Aq * Aq   +  CKernelAqCq[n,m,i,j] * Aq * Cq   +  CKernelAqDA[n,m,i,j] * Aq * DA  +  CKernelAqDC[n,m,i,j] * Aq * DC   
                        +  CKernelAqt4[n,m,i,j] * Aq * t4   +  CKernelBqDB[n,m,i,j] * Bq * DB   +  CKernelCqCq[n,m,i,j] * Cq * Cq  +  CKernelCqDA[n,m,i,j] * Cq * DA   
                        +  CKernelCqDC[n,m,i,j] * Cq * DC)
        end
        sum_mA += wq4_array[m] * inte_array_cmplx(KerA, wqv_array)/(pv*pv);
        sum_mB += wq4_array[m] * inte_array_cmplx(KerB, wqv_array);
        sum_mC += wq4_array[m] * inte_array_cmplx(KerC, wqv_array)/(p4*p4);
      end
      FUNC[(n-1) * 3 * N_PV + i]            = Ap_matrix[i,n]- 1    - sum_mA ;
      FUNC[(n-1) * 3 * N_PV + N_PV + i]     = Bp_matrix[i,n]- Mass - sum_mB ;
      FUNC[(n-1) * 3 * N_PV + 2 * N_PV + i] = Cp_matrix[i,n]- 1    - sum_mC;
    end
  end

end

end #end module
