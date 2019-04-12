using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef

function gaussianDensity(X)
	(n,d) = size(X)
        
	mu = (1/n)sum(X,dims=1)'
    #display(length(mu))
	Xc = X - repeat(mu',n)
	Sigma = (1/n)*(Xc'Xc)
	SigmaInv = Sigma^-1
    
    Sigma_0 = [(1/n)*(Xc'Xc), (1/n)*(Xc'Xc), (1/n)*(Xc'Xc)]
    
    display((d/2)log(2pi) + (1/2)logdet(Sigma_0[1]))
    
    function PDF(Xhat)
        #display(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)

		logZ = (d/2)log(2pi) + (1/2)logdet(Sigma)  
		for i in 1:t
			xc = Xhat[i,:] - mu
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[i] = exp(loglik)
		end
        #display(PDFs)
		return PDFs
	end
	return DensityModel(PDF)
end

function gaussianMixtureDensity(X)
    (n,d) = size(X)
        
	mu = (1/n)sum(X,dims=1)'
    #display(mu)
	Xc = X - repeat(mu',n)
	Sigma = (1/n)*(Xc'Xc)
	SigmaInv = Sigma^-1
    
    #Initializaion Stuff
    theta_0 = [1/3 1/3 1/3]
    #display(theta_0[1])
    mu_0 = [(1/n)sum(X[1:83,:],dims=1)', (1/n)sum(X[84:166,:],dims=1)', (1/n)sum(X[167:250,:],dims=1)']
    #display(mu_0)
    Sigma_0 = [(1/n)*(Xc'Xc), (1/n)*(Xc'Xc), (1/n)*(Xc'Xc)]
    
    #Recurring computations 
    function PofXgZ(X,mu_c,Sigma_c)
        (n_tmp,d_tmp)=size(X)
        prob = zeros(n_tmp,3)
        logZ = zeros(3)
        loglik = zeros(3)
        
        for c in 1:length(logZ)
            logZ[c] = (d_tmp/2)log(2pi) + (1/2)logdet(Sigma_c[c])                  
        end
        for c in 1:3
            for i in 1:n_tmp
                xc = X[i,:] - mu_c[c]
                loglik = -(1/2)dot(xc,(Sigma_c[c]^-1)*xc) - logZ[c]
                prob[i,c] = exp(loglik)    
            end
        end
        return prob
    end

    prob = PofXgZ(X,mu_0,Sigma_0)
    #display(theta)
    
    function respo(prob,theta)
        resp = zeros(n,3)
        for c in 1:3
            for i in 1:n
                resp[i,c] = (prob[i,c]*theta[c])/(sum(prob[i,1].*theta[1].+prob[i,2].*theta[2].+prob[i,3].*theta[3]))
            end
        end
        return resp
    end
    
    #Optimization EM Parameters
    prob = PofXgZ(X,mu_0,Sigma_0)
    #display(theta_0)
    resp = respo(prob,theta_0)
    #display(prob)
    #display(theta_0)
    #display(resp)
    
    theta_c = copy(theta_0)
    mu_c = copy(mu_0)
    Sigma_c = copy(Sigma_0)
    
    #display(mu_c[1])
    
    function updateParams(resp)
        theta_up = copy(theta_0)
        mu_up = copy(mu_0)
        Sigma_up = copy(Sigma_0)
        #display(Sigma[1])
        for c in 1:3
            theta_up[c] = (1/n)*sum(resp[:,c])
            mu_up[c] = (sum(resp[:,c].*X,dims=1)/sum(resp[:,c]))'
        end
        #display(mu_up)
        for c in 1:3
            Sigma_up[c] = (resp[:,c].*(X-repeat(mu_up[c]',n)))'*(X-repeat(mu_up[c]',n))/(sum(resp[:,c]))
        end
        return (theta_up,mu_up,Sigma_up)
    end
    
    (theta_c,mu_c,Sigma_c) = updateParams(resp)
    
    function loglikelihood(theta_c,mu_c,Sigma_c)
        prob_each = PofXgZ(X,mu_c,Sigma_c)
        for c in 1:3
            prob_each[:,c] .= theta_c[c]*prob_each[:,c]
        end
        #display(prob_each)

        sum_c_prob = sum(prob_each,dims=2)
        log_sum_c_prob = log.(sum_c_prob)
        loglik = sum(log_sum_c_prob)
        return loglik
    end
    
    #display(prob)
    #display(theta_c)
    (resp_c) = respo(prob,theta_0)
    #display(resp_c)
    
    loglik_c = loglikelihood(theta_c,mu_c,Sigma_c)
    
    display(loglik_c)
    
    
    #loop
    epsilon = 0.0001
    for i in 1:5000
        loglik_c_orig = copy(loglik_c)
        prob_c = PofXgZ(X,mu_c,Sigma_c)
        #display(prob_c)
        resp_c = respo(prob_c,theta_c) 
        (theta_c,mu_c,Sigma_c) = updateParams(resp_c)
        loglik_c = loglikelihood(theta_c,mu_c,Sigma_c)
        if abs(loglik_c-loglik_c_orig)<epsilon
            break
        end
    end
    
    display(loglik_c)
    
    function PDF(Xhat)
        #display(Xhat)
		(t,d) = size(Xhat)
		PDFs = zeros(t)
        
        prob_hat = PofXgZ(Xhat,mu_c,Sigma_c)
        for c in 1:3
            prob_hat[:,c] .= theta_c[c]*prob_hat[:,c]
        end
        #display(prob_each)
        sum_c_prob = sum(prob_hat,dims=2)
        PDFs = sum_c_prob
		return PDFs
	end
	return DensityModel(PDF)
end  
