function AllocateCommonVariables_Cartesian(N_PV, N_QV, N_P4, N_PHI,
								LambdaR_P4::T,LambdaV_P4::T,LambdaR_PV::T,LambdaV_PV::T)where{T}

  pv, wpv = GauLeg_Log(LambdaR_PV, LambdaV_PV, N_PV)
  qv, wqv = GauLeg_Log(LambdaR_PV, LambdaV_PV, N_QV)
  p4, wp4 = GauLeg_Log(LambdaR_P4, LambdaV_P4, N_P4)

  N_Q4 = 2 * N_P4
  q4 = zeros(N_Q4)
  wq4 = zeros(N_Q4)

  Threads.@threads for m = 1:N_P4
    q4[m+N_P4] = p4[m]
    q4[N_P4+1-m] = -p4[m]

    wq4[m+N_P4] = wp4[m]
    wq4[N_P4+1-m] = wp4[m]
  end

  zphi, wphi = GauLeg(-1.0, 1.0, N_PHI)

  Ap = zeros(Complex{T}, N_PV, N_P4)
  Bp = zeros(Complex{T}, N_PV, N_P4)
  Cp = zeros(Complex{T}, N_PV, N_P4)

  Aq = zeros(Complex{T}, N_QV, N_Q4)
  Bq = zeros(Complex{T}, N_QV, N_Q4)
  Cq = zeros(Complex{T}, N_QV, N_Q4)
  ABC_init = zeros(Complex{T}, 3 * N_P4 * N_PV)

  return N_Q4,pv,wpv,qv,wqv,p4,wp4,q4,wq4,zphi,wphi,Ap,Bp,Cp,Aq,Bq,Cq,ABC_init
end

function AllocateCommonVariables_Polar(N_PP, N_QQ, Np_PSI, N_PHI,
								LambdaR::T,LambdaV::T)where{T}

  pp, wpp = GauLeg_Log(LambdaR, LambdaV, N_PP)
  qq, wqq = GauLeg_Log(LambdaR, LambdaV, N_QQ)
  psi_p, wpsi_p = GauLeg(0.0::T, 1.0::T, Np_PSI)

  Nq_PSI = 2 * Np_PSI
  psi_q = zeros(Nq_PSI)
  wpsi_q = zeros(Nq_PSI)

  Threads.@threads for m = 1:Np_PSI
    psi_q[m+Np_PSI] =   psi_p[m]
    psi_q[Np_PSI+1-m] = - psi_p[m]

    wpsi_q[m+Np_PSI]   = wpsi_p[m]
    wpsi_q[Np_PSI+1-m] = wpsi_p[m]
  end

  zphi, wphi = GauLeg(-1.0, 1.0, N_PHI)

  Ap = zeros(Complex{T}, N_PP, Np_PSI)
  Bp = zeros(Complex{T}, N_PP, Np_PSI)
  Cp = zeros(Complex{T}, N_PP, Np_PSI)

  Aq = zeros(Complex{T}, N_QQ, Nq_PSI)
  Bq = zeros(Complex{T}, N_QQ, Nq_PSI)
  Cq = zeros(Complex{T}, N_QQ, Nq_PSI)
  ABC_init = zeros(Complex{T}, 3 * N_PP * Np_PSI)

  return Nq_PSI,pp,wpp,qq,wqq,psi_p,wpsi_p,psi_q,wpsi_q,zphi,wphi,Ap,Bp,Cp,Aq,Bq,Cq,ABC_init
end