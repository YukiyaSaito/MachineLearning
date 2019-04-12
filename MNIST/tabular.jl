include("misc.jl")
include("findMin.jl")

function tabular(X,y)
	(n,d) = size(X)

	# Compute the frequencies in the training data
	# (The below is *not* efficient in time or space)
	D =  Dict()
	for i in 1:n
		key = [y[i];X[i,:]]
		if haskey(D,key)
			D[key] += 1
		else
			D[key] = 1
		end
	end

	# Sample function
	function sampleFunc(xtilde)
		key0 = [0;xtilde]
		key1 = [1;xtilde]
		p0 = 0
		if haskey(D,key0)
			p0 = D[key0]
		end
		p1 = 0
		if haskey(D,key1)
			p1 = D[key1]
		end
		
		if p0+p1 == 0
			# Probability is undefined, go random
			return rand() < .5
		else
			return rand() < p1/(p0+p1)
		end
	end
	# Return model
	return SampleModel(sampleFunc)
end


function tabularDAG(X)
    m = size(X,1)
    n = size(X,3)

	# Compute the frequencies in the training data
	# (The below is *not* efficient in time or space)
    
	D =  Dict()
    Num = Dict()
    ###
    Threads.@threads for ex in 1:n 
        for i in 1:m
            for j in 1:m
                if i==1 && j==1
                    key = [i j 0 0 0 0 0 0 0 0]
                elseif i==1 && j==2
                    key = [i j 0 0 0 0 0 X[i,j-1,ex] 0 0]
                elseif i==1 && j>2
                    key = [i j 0 0 X[i,j-2,ex] 0 0 X[i,j-1,ex] 0 0]
                elseif i==2 && j==1
                    key = [i j 0 0 0 0 0 0 0 X[i-1,j,ex]] 
                elseif i==2 && j==2
                    key = [i j 0 0 0 0 X[i-1,j-1,ex] X[i,j-1,ex] 0 X[i-1,j,ex]] 
                elseif i==2 && j>2
                    key = [i j 0 0 0 X[i-1,j-2,ex] X[i-1,j-1,ex] X[i-1,j,ex] X[i,j-2,ex] X[i,j-1,ex]]
                elseif i>2 && j==1
                    key = [i j 0 0 X[i-2,j,ex] 0 0 X[i-1,j,ex] 0 0]
                elseif i>2 && j==2
                    key = [i j 0 X[i-2,j-1,ex] X[i-2,j,ex] 0 X[i-1,j-1,ex] X[i-1,j,ex] 0 X[i,j-1,ex]]
                else
                    key = [i j X[i-2,j-2,ex] X[i-2,j-1,ex] X[i-2,j,ex] X[i-1,j-2,ex] X[i-1,j-1,ex] X[i-1,j,ex] X[i,j-2,ex] X[i,j-1,ex]]
                end
                if haskey(D,key)
                    D[key] += 1
                else
                    D[key] = 1
                end
                if haskey(Num,key)
                    if X[i,j,ex]==1
                        Num[key] += 1
                    end
                else
                    if X[i,j,ex]==1
                        Num[key] = 1
                    else
                        Num[key] = 0
                    end
                end
            end
        end
    end
    ###
    return(D,Num)
end
