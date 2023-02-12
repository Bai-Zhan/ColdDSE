module DSEMathWrapper

export inte_array_cmplx

"""
对于数组fun[],作为函数在其多项式零点上的值，并已知积分权重，给出积分值
"""
function inte_array_cmplx(func, w) 
  s = zero(eltype(func))
  for i = eachindex(func)
    s += func[i] * w[i]
  end
  return s
end

end # module
