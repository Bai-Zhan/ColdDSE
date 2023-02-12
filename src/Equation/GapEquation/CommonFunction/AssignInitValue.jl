#include(joinpath(@__DIR__,"TransformABC.jl"))

function AssignInitValue!(init_value, Mass, N1, N2, Ap, Bp, Cp)
  if init_value == "W" 
    for i = 1:N1, n = 1:N2
      Ap[i,n] = 1
      Bp[i,n] = Mass
      Cp[i,n] = 1
    end
  elseif init_value == "N" 
    for i = 1:N1, n = 1:N2
      Ap[i,n] = 2
      Bp[i,n] = 2
      Cp[i,n] = 2
    end
  else
    println(init_value)
    println("wrong init value!")
  end
end

function AssignInitValue!(init_value, Mass, N1, N2, A, B, C,ABC)
	AssignInitValue!(init_value, Mass, N1, N2, A, B, C)
	CombineABC!(N1,N2,A,B,C,ABC)
end