function SolveGapEquation(::Type{T},constants;result_file)where{T<:Real}
	if constants.vertex=="bare"
		if constants.discretization == "cartesian"
    		BareVertexCartesian.solve(T,constants;result_file)
		elseif constants.discretization=="polar"
    		BareVertexPolar.solve(T,constants;result_file)
		else
			print("wrong discretization!",@__FILE__,@__LINE__)
		end
	elseif constants.vertex=="CLR"
		if constants.discretization == "cartesian"
    		CLRVertexCartesian.solve(T,constants;result_file="")
		else
			print("wrong discretization!",@__FILE__,@__LINE__)
		end
	else
		print("wrong vertex!")
	end

end