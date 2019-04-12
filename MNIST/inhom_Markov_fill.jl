# Load X and y variable
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

# Train an independent Bernoulli model
p_ij00 = zeros(m,m)
p_ij01 = zeros(m,m) #redundant
p_ij10 = zeros(m,m)
p_ij11 = zeros(m,m) #redundant
for j in 1:m
    tempmat = X[:,j,:] #should be m x n matrix
    for i in 2:m
        if sum(tempmat[i-1,:] .== 0) !=0 && sum(tempmat[i-1,:] .== 0) !=0
            p_ij00[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 0].==0)/sum(tempmat[i-1,:] .== 0)
            p_ij01[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 0].==1)/sum(tempmat[i-1,:] .== 0)
            p_ij10[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 1].==0)/sum(tempmat[i-1,:] .== 1)
            p_ij11[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 1].==1)/sum(tempmat[i-1,:] .== 1)
        elseif sum(tempmat[i-1,:] .== 0) ==0 && sum(tempmat[i-1,:] .== 0) !=0
            p_ij00[i,j] = 0
            p_ij01[i,j] = 0
            p_ij10[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 1].==0)/sum(tempmat[i-1,:] .== 1)
            p_ij11[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 1].==1)/sum(tempmat[i-1,:] .== 1)
        elseif sum(tempmat[i-1,:] .== 0) !=0 && sum(tempmat[i-1,:] .== 0) ==0
            p_ij00[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 0].==0)/sum(tempmat[i-1,:] .== 0)
            p_ij01[i,j] = sum(tempmat[i,tempmat[i-1,:] .== 0].==1)/sum(tempmat[i-1,:] .== 0)
            p_ij10[i,j] = 0
            p_ij11[i,j] = 0
        elseif sum(tempmat[i-1,:] .== 0) ==0 && sum(tempmat[i-1,:] .== 0) ==0
            p_ij00[i,j] = 0
            p_ij01[i,j] = 0
            p_ij10[i,j] = 0
            p_ij11[i,j] = 0
        end
    end
end

# Show Bernoulli parameters
figure(1)
imshow(p_ij00)
figure(2)
imshow(p_ij01)
figure(3)
imshow(p_ij10)
figure(4)
imshow(p_ij11)

# Fill-in some random test images
t = size(Xtest,3)
figure(5)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                if I[i-1,j] == 0
                    I[i,j] = rand() < p_ij01[i,j]
                elseif I[i-1,j] == 1
                    I[i,j] = rand() < p_ij11[i,j]
                end
            end
        end
    end
    imshow(I)
end
