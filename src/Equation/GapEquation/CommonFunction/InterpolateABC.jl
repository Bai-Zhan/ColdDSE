using Dierckx
function InterpolateABC!(N_PV::Integer, N_QV::Integer, N_P4::Integer, N_Q4::Integer,
  pv::Vector{T}, qv::Vector{T},
  Ap::Array{Complex{T}}, Bp::Array{Complex{T}}, Cp::Array{Complex{T}},
  Aq::Array{Complex{T}}, Bq::Array{Complex{T}}, Cq::Array{Complex{T}}) where {T<:AbstractFloat}

  real_A_temp = zeros(T, N_PV)
  real_B_temp = zeros(T, N_PV)
  real_C_temp = zeros(T, N_PV)

  imag_A_temp = zeros(T, N_PV)
  imag_B_temp = zeros(T, N_PV)
  imag_C_temp = zeros(T, N_PV)

  for n = 1:N_P4
    for i = 1:N_PV
      real_A_temp[i] = real(Ap[i,n])
      real_B_temp[i] = real(Bp[i,n])
      real_C_temp[i] = real(Cp[i,n])

      imag_A_temp[i] = imag(Ap[i,n])
      imag_B_temp[i] = imag(Bp[i,n])
      imag_C_temp[i] = imag(Cp[i,n])
    end

    spline_realA = Spline1D(pv, real_A_temp)
    spline_realB = Spline1D(pv, real_B_temp)
    spline_realC = Spline1D(pv, real_C_temp)

    spline_imagA = Spline1D(pv, imag_A_temp)
    spline_imagB = Spline1D(pv, imag_B_temp)
    spline_imagC = Spline1D(pv, imag_C_temp)


    for j = 1:N_QV
      Aq[j,n+N_P4] = complex(spline_realA(qv[j]), spline_imagA(qv[j]))
      Bq[j,n+N_P4] = complex(spline_realB(qv[j]), spline_imagB(qv[j]))
      Cq[j,n+N_P4] = complex(spline_realC(qv[j]), spline_imagC(qv[j]))

      Aq[j,N_P4-n+1] = conj(Aq[j,n+N_P4])
      Bq[j,N_P4-n+1] = conj(Bq[j,n+N_P4])
      Cq[j,N_P4-n+1] = conj(Cq[j,n+N_P4])
    end
  end
end