# Load X and y variable
include("logReg.jl")
using JLD, PyPlot
using LinearAlgebra
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)

display(n)

Wdictionary = logRegDAG(X)

#display(Wdictionary)

# Fill-in some random test images
t = size(Xtest,3)
figure(1)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]
    XtestVec = I'[:]
    #display(XtestVec[393:764])
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            key = [i j]
            if isnan(I[i,j])
                #display(XtestVec[1:((i-1)*m + (j-1))])
                wij = Wdictionary[key]
                rangeInd = (i-1)*m + (j-1)
                #display(dot([1;XtestVec[1:rangeInd]],wij[:]))
                #display(Wdictionary[key])
                I[i,j] = rand() < 1 ./ (1+exp(-dot([1;XtestVec[1:rangeInd]],wij)))
                XtestVec[rangeInd+1]=I[i,j]
            end
        end
    end
    #display(I[:])
    imshow(I)
end
