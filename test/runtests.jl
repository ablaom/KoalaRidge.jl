using Koala
using KoalaRidge
using Base.Test

# Load some data and define train/test rows:
const X, y = load_ames();
y = log.(y)

const train, test = splitrows(eachindex(y), 0.7); # 70:30 split

# Instantiate a model:
ridge = RidgeRegressor(standardize=true,
                              boxcox_inputs=true, lambda=0.1)
showall(ridge)

# Build a machine (excuding :YearRemodAdd):
ridgeM = SupervisedMachine(ridge, X, y, train)
showall(ridgeM)

fit!(ridgeM, train)
showall(ridgeM)

lambdas = logspace(-3,1,100)

# tune using cross-validation
lambdas, rmserrors = @curve λ lambdas begin
    ridge.lambda = λ
    mean(cv(ridgeM, eachindex(train), parallel=true, n_folds=9, verbosity=0))
end

ridge.lambda = lambdas[indmin(rmserrors)]

fit!(ridgeM, train)

score = err(ridgeM, test)
println("score = $score")
@test score > 0.12 && score < 0.14
