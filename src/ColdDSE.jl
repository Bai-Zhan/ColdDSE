module ColdDSE
  export SolveGapEquation

  import GluonModel
  include(joinpath(@__DIR__,"Equation","GapEquation","BareVertexCartesian.jl"))
  include(joinpath(@__DIR__,"Equation","GapEquation","CLRVertexCartesian.jl"))
  include(joinpath(@__DIR__,"Equation","GapEquation","BareVertexPolar.jl"))
  include(joinpath(@__DIR__,"Equation","GapEquation","SolveGapEquation.jl"))
  include(joinpath(@__DIR__,"Constants.jl"))

  import .BareVertexCartesian
  import .CLRVertexCartesian
  import .BareVertexPolar

  function test()
    println("Loading module ColdDSE...")
    #GluonModel.test()

    constants=GapEquationConstants(;N_QV=19)
    SolveGapEquation(Float64,constants,result_file="")
    #BareVertexCartesian.solve(Float64,0.005,0.1; init_type="N", gluon_model="gauss", Gluon_D_Long=1.0,Gluon_D_Tran=1.0, Gluon_omega=0.5, LambdaR_P4=1e-2, LambdaV_P4=1e2, LambdaR_PV=1e-2, LambdaV_PV=1e2, N_PV=20, N_QV=20, N_P4=25, N_PHI=30, result_file="")
    #BareVertexCartesian.solve(Float64,constants,result_file="")
    #BareVertexPolar.solve(Float64, 0.005, 0.1; 
    #            init_type="N",gluon_model="gauss",
    #            Gluon_D_Long=1.0, Gluon_D_Tran=1.0, Gluon_omega=0.5, LambdaR=1e-2, LambdaV=1e2, 
    #            N_PP=20, N_QQ=20, Np_PSI=20, N_PHI=30,
    #            result_file="")
    #BareVertexPolar.solve(Float64,constants,result_file="")
    #CLRVertexCartesian.solve(Float64,constants,result_file="")
    #print(CreateResultPath(false,constants))
  end

end # module ColdDSE