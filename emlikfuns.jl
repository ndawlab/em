# julia EM model fitting, Nathaniel Daw 8/2019
# example likelihood functions

function qlik(params,data)
	# basic rescorla wagner learning
	beta = params[1]
	lr = 0.5 + 0.5 * erf(params[2] / sqrt(2)) # (this is unit normal cdf
 											  # so squashes any input to 0-1 and 
 											  # transforms unit normal prior to uniform prior)

	# It is important that we specify the type of Q below, as corresponding to the type 
	# of the input parameters
	# This allows the function to work with the algebraic differentiation 
	# the principle ("type stability") is that any variable that holds a function of the input params
	# (here Q and ultimately lik)
	# has to be of the same type as the input 

	Q = zeros(typeof(beta),2)  

	lik = 0.
	
	r::Array{Int64,1} = data.r
	c::Array{Int64,1} = data.c
	
	for i = 1:length(c)
		if (c[i]>0)
    
			lik += beta*Q[c[i]] - logsumexp(beta*Q)
	
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

	Q = zeros(typeof(beta),2)  

	lr = convert(typeof(beta),1);

	lik = 0.
	
	r::Array{Int64,1} = data.r
	c::Array{Int64,1} = data.c

	for i = 1:length(c)
		if (c[i]>0)
    
			lik += beta*Q[c[i]] - logsumexp(beta*Q)
	
			delta = r[i] - Q[c[i]]
			Q[c[i]] +=  lr/2 * delta
			lr = (1-nu) * lr + nu * abs(delta)
		end
	end
	return -lik
end

function seqlik(params,data)
	# this is more or less the likelihood function for the two-step decision task from Gillan et al., eLife 2015

	beta1m = params[1]
	beta1t0 = params[2]
	beta1t1 = params[3]
	beta2 = params[4]
 	lr = 0.5 + 0.5 * erf(params[5] / sqrt(2))  
	ps = params[6]

	c1::Array{Int64,1} = data.ch1
	c2::Array{Int64,1} = data.ch2
	r::Array{Int64,1} = data.mn
	s::Array{Int64,1} = data.st

	Q0 = zeros(typeof(beta1m),3,2)
	Q1 = zeros(typeof(beta1m),2)
	Qm = zeros(typeof(beta1m),2)
	

	lik = 0.

	prevc = 0

	for i = 1:length(c1)
		if (c1[i]>0)
			# this uses a differentiable approximation to the max over the two actions at each state
			Qm = [softmaximum(Q0[2,1],Q0[2,2]),softmaximum(Q0[3,1],Q0[3,2])] 
			Qd = beta1m * Qm + beta1t0 * vec(Q0[1,:]) + beta1t1 * Q1
			
			if prevc>0
				Qd[prevc] += ps
			end
    
			lik += Qd[c1[i]] - logsumexp(Qd)
    
			if (c2[i]>0)
				lik += beta2 * Q0[s[i],c2[i]] - logsumexp(beta2 * Q0[s[i],:])
				
				Q0[1,c1[i]] = (1-lr) * Q0[1,c1[i]] + Q0[s[i],c2[i]]
				Q1[c1[i]] = (1-lr) * Q1[c1[i]] + r[i]	
				Q0[s[i],c2[i]] = (1-lr) * Q0[s[i],c2[i]] + r[i]
			end		
		end
		prevc = c1[i]
	end
	return -lik
end


####
#### simulate a dataset corresponding to the above 3 likelihood functions

function simq(params, ntrials)
	beta = params[1]
	#lr = 1/(1+exp(-params[2]))
	lr = 0.5 + 0.5 * erf(params[2] / sqrt(2))
	
	c = zeros(Int64,ntrials)
	r = zeros(Int64,ntrials)
	
	Q = zeros(2)

	for i = 1:ntrials
    
		ps = exp.(beta*Q)
		ps = ps / sum(ps)
		c[i] = (rand() > ps[1]) + 1
			
		r[i] = 2*round(rand()) - 1
		Q[c[i]] = (1-lr) * Q[c[i]] + r[i]
	end	
	return (c,r)
end

function simseq(params, ntrials)
	beta1m = params[1]
	beta1t0 = params[2]
	beta1t1 = params[3]
	beta2 = params[4]
 	lr = 0.5 + 0.5 * erf(params[5] / sqrt(2))
	ps = params[6]

	c1 = zeros(Int64,ntrials)
	c2 = zeros(Int64,ntrials)
	r = zeros(Int64,ntrials)
	s = zeros(Int64,ntrials)

	Q0 = zeros(typeof(beta1m),3,2)
	Q1 = zeros(typeof(beta1m),2)
	Qm = zeros(typeof(beta1m),2)
	
	prevc = 0
	
	for i = 1:ntrials
		Qm = [softmaximum(Q0[2,1],Q0[2,2]),softmaximum(Q0[3,1],Q0[3,2])]
		Qd = beta1m * Qm + beta1t0 * vec(Q0[1,:]) + beta1t1 * Q1
			
		if prevc>0
			Qd[prevc] += ps
		end
    
		cp = exp.(beta2*Qd)
		cp = cp / sum(cp)
		c1[i] = (rand() > cp[1]) + 1
		s[i] = (rand() > ( c1[i] == 1 ? .7 : .3  )) + 2

		cp = exp.(beta2*Q0[s[i],:])
		cp = cp / sum(cp)
		c2[i] = (rand() > cp[1]) + 1
			
		r[i] = 2*round(rand()) - 1

		Q0[1,c1[i]] = (1-lr) * Q0[1,c1[i]] + Q0[s[i],c2[i]]
		Q1[c1[i]] = (1-lr) * Q1[c1[i]] + r[i]	
		Q0[s[i],c2[i]] = (1-lr) * Q0[s[i],c2[i]] + r[i]

		prevc = c1[i]
	end
	return (c1,s,c2,r)
end

function simjian(params, ntrials)
	beta = 2 * params[1]
	nu = 0.5 + 0.5 * erf(params[2] / sqrt(2))
	
	c = zeros(Int64,ntrials)
	r = zeros(Int64,ntrials)
	
	Q = zeros(2)
	lr = 1

	for i = 1:ntrials
    
		ps = exp(beta*Q)
		ps = ps / sum(ps)
		c[i] = (rand() > ps[1]) + 1
			
		r[i] = 2*round(rand()) - 1
		delta = r[i] - Q[c[i]]
		Q[c[i]] +=  lr/2 * delta
		lr = (1-nu) * lr + nu * abs(delta)
	end	
	return (c,r)
end
