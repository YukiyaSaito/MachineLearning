using JLD, PyPlot

# Load multi-modal data
data = load("mixtureData.jld")
X = data["X"]

include("gaussianDensity.jl")
model = gaussianMixtureDensity(X)

# Plot data and densities (you can ignore the code below)
plot(X[:,1],X[:,2],".")

increment = 100
(xmin,xmax) = xlim()
xDomain = range(xmin,stop=xmax,length=increment)
(ymin,ymax) = ylim()
yDomain = range(ymin,stop=ymax,length=increment)

xValues = repeat(xDomain,1,length(xDomain))
yValues = repeat(yDomain',length(yDomain),1)

z = model.pdf([xValues[:] yValues[:]])


@assert(length(z) == length(xValues),"Size of model function's output is wrong");

zValues = reshape(z,size(xValues))
#display(zValues)

contour(xValues,yValues,zValues)
