using Parameters
using Printf

@with_kw mutable struct GapEquationConstants{T<:Real}
  vertex::String = "bare"
  gluon_model::String = "gauss"
  init_type::String="N"
  discretization::String = "cartesian"
  
  N_PP::Integer           =20   #The point number for the modulus of 4-momentum p
  N_QQ::Integer           =20   #The point number for the modulus of 4-momentum q
  N_theta::Integer        =20   #The angle between the 4-momentum p and q
  Np_PSI::Integer         =20   #The angle between 4-momentum p and \vec{p}
  
  N_PV 	::Integer         =20   #有限温或有限密时，三动量模的取点数
  N_P4 	::Integer         =20   #零温有限密时，第四动量的取点数
  N_QV 	::Integer         =20   #有限温或有限密时，积分内三动量的取点数
  N_PHI	::Integer         =30   #The angle between 3-momentum \vec{p} and \vec{p}
  
  Gluon_D::T              =1.0  #GeV**2
  Gluon_omega::T          =0.5  #GeV
  Gluon_D_Tran::T         =1.0
  Gluon_D_Long::T         =1.0
  Gluon_mass::T           =0.05 # Gluon thermal mass in the unit of GeV
  
  pole_handle_s::T   =1e-5
  # For some of the vertices, there will be a pole at p=q, and we need to exclude this pole from integral.
  # This parameter determines the range of the exclusion. 
  # The larger this parameter is, the larger neighbourhood of the pole will be excluded ( and the accuracy of the solution is worse).
  
  
  LambdaR::T              =1e-2  # infrared cut-off
  LambdaV::T              =1e2   # ultra-violet cut-off
  LambdaR_PV::T           =1e-2
  LambdaV_PV::T           =1e2
  LambdaR_P4::T           =1e-2
  LambdaV_P4::T           =1e2
  
  mass::T                 =0.005 # current quark mass
  mu::T                   =0.1   # chemical potential

  eta::T=-0.65#CLR顶点和tau5顶点的参数
  
  #int ReScheme=1;
  #// 1 current quark mass independent scheme used by Chen Jing. A(\mu)=1, \frac{dB(p)}{dm_{f}}|_{p=\mu}=1, 
  #// 2 current quark mass dependent scheme used by Tang Can.  S^{-1}(\mu)=i\slash{\mu}+m_{f}
  #//   Z1=Z2^2 for RL vertex, Z1=Z2 for TGL vertex.
  #// 3 current quark mass dependent scheme used by Qin Si-Xue. S^{-1}(\mu)=i\slash{\mu}+m_{f}
  #//   Z1 is absorbed in the gluon model.
  #
  #double RePoint=19.0;//重整化点，取零为不重整
  #
  #double INIT_Z_2A=1.0,INIT_Z_2C=1.0,INIT_Z_4=1.0,INIT_Z_M;
  #double Z2_d1m=0,ZM_d1m=0;
  #double Z2_d2m=0,ZM_d2m=0;
  #//no renormalization
  #
  #
  #//求解未做重整化的DS方程时的假设的重整化常数
  #double Z_2A0=1,Z_2C0=1,Z_2M0,Z_40=1;//重整化常数,Z_2M0代表裸质量与场强重整化常数的乘积
  #double Z_2A0_cmplx=1,Z_2C0_cmplx=1,Z_2M0_cmplx=1,Z_40_cmplx=1;//重整化常数,计算复平面DS方程时使用
  #
  #
  #double ALPHA=0.4413;
  #double ALPHA_T=0.0;
  #double MU_C=0.3076;
  #double M_PION=0.14;
end

function CreateResultPath(is_create::Bool,constants;result_folder="./result")

  #	Layer 1: vertex and gluon
  path = @sprintf("%s/%s_%s",result_folder,constants.vertex,constants.gluon_model)

  #	Layer 2：parameter in vertex and gluon model
  path =@sprintf("%s/DL%.6f_DT%.6f_omega%.6f",path,constants.Gluon_D_Long,constants.Gluon_D_Tran,constants.Gluon_omega)

  #if constants.vertex=="BC" || constants.vertex=="CLR"
  #  path =@sprintf("%s_eta%.6f_poles%.3e",path,constants.eta,constants.pole_handle_s)
  #end


  #	layer 3: renormalization and current quark mass
  path =@sprintf("%s/Mass%.6f",path,constants.mass)

  #	layer 4: chemical potential
  path =@sprintf("%s/mu%.6f",path,constants.mu)

  mkpath(path)
  return path
end
