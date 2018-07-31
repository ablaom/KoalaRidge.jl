using Revise
using Koala
using KoalaRidge
using Base.Test
using DataFrames

## SYNTHETIC DATA TEST

# Define some linear, noise-free, synthetic data:
const bias = -42.0
const coefficients = Float64[1, 3, 7]
const A = randn(1000, 3)
const X = DataFrame(A)
const y = A*coefficients

# Train model on all data with no regularization and no
# standardization of target:
const ridge = RidgeRegressor(lambda=0.0)
ty = default_transformer_y(ridge)
ty.standardize = false
ridgeM = Machine(ridge, X, y, eachindex(y), transformer_y=ty)
fit!(ridgeM, eachindex(y))

# Training error:
E = err(ridgeM, eachindex(y))
@test E < 1e-10

# Get the true bias?
@test abs(ridgeM.predictor.bias) < 1e-10
@test norm(ridgeM.predictor.coefficients - coefficients) < 1e-10

# Train with standardization of target:
ridgeM1 = Machine(ridge, X, y, eachindex(y))
fit!(ridgeM1, eachindex(y))

# Get same predictions?
@test isapprox(predict(ridgeM, X), predict(ridgeM1, X))


## TEST OF OTHER METHODS ON REAL DATA

# Load some data and define train/test rows:
const X, y = load_ames();
y = log.(y) # log of the SalePrice
const train, test = split(eachindex(y), 0.7); # 70:30 split

# Instantiate a model:
ridge = RidgeRegressor(lambda=0.1)
ty = default_transformer_y(ridge)
ty.boxcox = false
showall(ty)
showall(ridge)

# Build a machine:
ridgeM = Machine(ridge, X, y, train, transformer_y=ty)

fit!(ridgeM, train)
showall(ridgeM)

# tune lambda using cross-validation
lambdas, rmserrors = @curve λ logspace(-3,1,100) begin
    ridge.lambda = λ
    mean(cv(ridgeM, train, n_folds=9, verbosity=0))
end

# set lambda to the optimal value and do final train:
ridge.lambda = lambdas[indmin(rmserrors)]
fit!(ridgeM, train)

score = err(ridgeM, test)
println("score = $score")
@test score > 0.12 && score < 0.14

