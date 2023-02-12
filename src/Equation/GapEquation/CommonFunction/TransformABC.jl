function CombineABC!(N1::Integer, N2::Integer, A, B, C, ABC)
  for i = 1:N1, n = 1:N2
    ABC[(n-1)*3*N1+i]      = A[i,n]
    ABC[(n-1)*3*N1+N1+i]   = B[i,n]
    ABC[(n-1)*3*N1+2*N1+i] = C[i,n]
  end
end

function SplitABC!(N1::Integer, N2::Integer, ABC, A, B, C)
  for i = 1:N1, n = 1:N2
      A[i,n] = ABC[(n-1)*3*N1+i]
      B[i,n] = ABC[(n-1)*3*N1+N1+i]
      C[i,n] = ABC[(n-1)*3*N1+2*N1+i]
  end
end