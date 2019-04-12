# Load X and y variable
include("tabular.jl")
using JLD, PyPlot
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

model = zeros(m,m)
(D,Num)=tabularDAG(X)


# Show Bernoulli parameters
#figure(1)
#imshow(p_ij)

# Fill-in some random test images
t = size(Xtest,3)
figure(1)
figure(2)
for image in 1:4
    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]
    figure(2)
    subplot(2,2,image)
    imshow(I)
    figure(1)
    subplot(2,2,image)
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
             if isnan(I[i,j])
                if j==1
                    key = [i j 0 0 I[i-2,j] 0 0 I[i-1,j] 0 0]
                elseif j==2
                    key = [i j 0 I[i-2,j-1] I[i-2,j] 0 I[i-1,j-1] I[i-1,j] 0 I[i,j-1]]
                elseif j>2
                    key = [i j I[i-2,j-2] I[i-2,j-1] I[i-2,j] I[i-1,j-2] I[i-1,j-1] I[i-1,j] I[i,j-2] I[i,j-1]]
                end
                if haskey(D,key)
                    model[i,j]=Num[key]/D[key]
                else
                    model[i,j]=0.5
                end
                I[i,j] = rand() < model[i,j]
            end
        end
    end
    imshow(I)
end
