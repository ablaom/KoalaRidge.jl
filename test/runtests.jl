using Koala
using KoalaRidge
using Base.Test

# Load some data and define train/test rows:
const X, y = load_ames();
y = log.(y) # log of the SalePrice

const train, test = splitrows(eachindex(y), 0.7); # 70:30 split

# Instantiate a model:
ridge = RidgeRegressor(lambda=0.1, standardize=true, boxcox_inputs=true)
showall(ridge)

# Build a machine:
ridgeM = SupervisedMachine(ridge, X, y, train)

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
