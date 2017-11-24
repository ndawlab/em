# julia EM model fitting, Nathaniel Daw 11/2015

function qlik(params,data)
	# basic rescorla wagner learning
	beta = params[1]
	lr = 0.5 + 0.5 * erf(params[2] / sqrt(2))

	# It is important that we specify the type of Q below, as corresponding to the type 
	# of the input parameters
	# This allows the function to work with the algebraic differentiation 
	# the principle ("type stability") is that any variable that holds a function of the input params
	# (here Q and ultimately lik)
	# has to be of the same type as the input 

	Q = zeros(typeof(beta),2)  

	lik = 0
	
	r = data[:r]
	c = data[:c]
	
	for i = 1:length(c)
		if (c[i]>0)
    
			lik += beta*Q[c[i]] - log(sum(exp.(beta*Q)))
	
			Q[c[i]] = (1-lr) * Q[c[i]] + r[i]
		end
	end

	return -lik
end

function jianlik(params,data)
	# this is a simplified form of the hybrid model from Li et al, Nat Neuro 2011
	# for comparison with qlik above

	beta = 2 * params[1] # match scaling with qlik
	nu = 0.5 + 0.5 * erf(params[2] / sqrt(2))

	# It is important that we specify the type of Q & lr below, as corresponding to the type 
	# of the input parameters
	# This allows the function to work with the algebraic differentiation 
	# the principle ("type stability") is that any variable that holds a function of the input params
	# (here Q, lr, and ultimately lik)
	# has to be of the same type as the input 

	Q = zeros(typeof(beta),2)  
