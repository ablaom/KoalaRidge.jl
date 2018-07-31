# ![logo](logo.png) KoalaRidge.jl

> A [Koala](https://github.com/ablaom/Koala.jl) wrap of the
[`MultivariateStats.jl`](https://github.com/JuliaStats/MultivariateStats.jl)
implementation of ridge regression.

## Basic usage

````julia
    julia> using Koala
    julia> using KoalaRidge
````

Load some data and define train/test rows:

````julia
    julia> const X, y = load_ames();
    julia> y = log.(y);  # log of the SalePrice
    julia> const train, test = splitrows(eachindex(y), 0.7); # 70:30 split
````

Instantiate a model:

````julia
    julia> ridge = RidgeRegressor(lambda=0.1, standardize=true, boxcox_inputs=true)
    RidgeRegressor@...402
	
    julia> showall(ridge)
    RidgeRegressor@...402
````

key                     | value
------------------------|------------------------
boxcox                  |false
boxcox_inputs           |true
drop_last               |true
lambda                  |0.1
shift                   |true
standardize             |true
	
Build a machine:

````julia
    julia> ridgeM = SupervisedMachine(ridge, X, y, train)
    INFO: Computing input Box-Cox transformations.
    Box-Cox transformations: 
     :OverallQual    (*not* transformed, less than 16 values)
     :GrLivArea    lambda=0.18  shift=0.0
     :x1stFlrSF    lambda=0.1  shift=0.0
     :TotalBsmtSF    lambda=0.96  shift=205.7576054955839
     :BsmtFinSF1    lambda=0.7  shift=84.40490677134446
     :LotArea    lambda=-0.02  shift=0.0
     :GarageCars    (*not* transformed, less than 16 values)
     :GarageArea    lambda=1.06  shift=92.25122669283611
     :YearRemodAdd    (*not* transformed, lambda too extreme)
     :YearBuilt    (*not* transformed, lambda too extreme)
    INFO: Determining one-hot encodings for inputs.
    Spawned 24 columns to hot-encode Neighborhood.
    Spawned 14 columns to hot-encode MSSubClass.
    INFO: Computing target standardization.
    SupervisedMachine{RidgeRegressor}@...817

    julia> fit!(ridgeM, train)
    SupervisedMachine{RidgeRegressor}@...817

    julia> showall(ridgeM)
	 SupervisedMachine{RidgeRegressor}@...817
````

key                     | value
------------------------|------------------------
Xt                      |Array{Float64,2} of shape (1456, 48)
model                   |RidgeRegressor@...402
n_iter                  |1
predictor               |LinearPredictor@...401
report                  |Dict with keys: Symbol[:feature_importance_curve]
scheme_X                |Scheme_X@...218
scheme_y                |Scheme_y@...087
yt                      |Array{Float64,1} of shape (1456,)

Model detail:
RidgeRegressor@...402

key                     | value
------------------------|------------------------
boxcox                  |false
boxcox_inputs           |true
drop_last               |true
lambda                  |0.1
shift                   |true
standardize             |true

````julia
                                 Feature importance (coefs of linear predictor)
                                 ┌────────────────────────────────────────┐ 
        Neighborhood__IDOTRR (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.529 │ 
             MSSubClass___85 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.399         │ 
            MSSubClass___180 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.397         │ 
       Neighborhood__SawyerW (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.357            │ 
       Neighborhood__Edwards (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.341             │ 
       Neighborhood__MeadowV (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.334             │ 
       Neighborhood__Mitchel (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.331             │ 
                     LotArea (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.329             │ 
       Neighborhood__OldTown (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.313              │ 
       Neighborhood__Gilbert (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.309               │ 
             MSSubClass___80 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.291                │ 
        Neighborhood__Sawyer (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.292                │ 
        Neighborhood__NWAmes (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.285                │ 
                   GrLivArea (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.28                  │ 
       Neighborhood__CollgCr (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.254                  │ 
             MSSubClass___20 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.252                  │ 
        Neighborhood__Timber (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.242                   │ 
       Neighborhood__Crawfor (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.229                    │ 
             MSSubClass___60 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.222                    │ 
         Neighborhood__NAmes (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.218                    │ 
             MSSubClass___45 (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.213                     │ 
       Neighborhood__StoneBr (+) │▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.209                     │ 
        Neighborhood__BrDale (-) │▪▪▪▪▪▪▪▪▪▪▪▪▪ 0.202                     │ 
                 OverallQual (+) │▪▪▪▪▪▪▪▪▪▪▪▪ 0.199                      │ 
             MSSubClass___30 (+) │▪▪▪▪▪▪▪▪▪▪▪▪ 0.185                      │ 
         Neighborhood__SWISU (-) │▪▪▪▪▪▪▪▪▪▪▪ 0.174                       │ 
            MSSubClass___120 (+) │▪▪▪▪▪▪▪▪▪▪▪ 0.17                        │ 
             MSSubClass___50 (+) │▪▪▪▪▪▪▪▪▪▪ 0.167                        │ 
             MSSubClass___75 (+) │▪▪▪▪▪▪▪▪▪ 0.152                         │ 
       Neighborhood__NoRidge (-) │▪▪▪▪▪▪▪▪▪ 0.15                          │ 
             MSSubClass___70 (+) │▪▪▪▪▪▪▪▪▪ 0.143                         │ 
             MSSubClass___40 (+) │▪▪▪▪▪▪▪▪▪ 0.137                         │ 
       Neighborhood__Blueste (+) │▪▪▪▪▪▪▪▪ 0.128                          │ 
       Neighborhood__Blmngtn (-) │▪▪▪▪▪▪▪ 0.114                           │ 
       Neighborhood__ClearCr (-) │▪▪▪▪▪▪ 0.095                            │ 
                  GarageCars (+) │▪▪▪▪▪▪ 0.093                            │ 
       Neighborhood__BrkSide (-) │▪▪▪▪▪ 0.076                             │ 
       Neighborhood__Somerst (-) │▪▪▪ 0.048                               │ 
                   x1stFlrSF (-) │▪▪ 0.035                                │ 
            MSSubClass___190 (+) │▪▪ 0.032                                │ 
       Neighborhood__NridgHt (+) │▪▪ 0.032                                │ 
       Neighborhood__NPkVill (+) │▪ 0.023                                 │ 
                YearRemodAdd (+) │ 0.006                                  │ 
            MSSubClass___160 (-) │ 0.006                                  │ 
                   YearBuilt (+) │ 0.004                                  │ 
                  BsmtFinSF1 (+) │ 0.001                                  │ 
                 TotalBsmtSF (+) │ 0.0                                    │ 
                  GarageArea (+) │ 0.0                                    │ 
                                 └────────────────────────────────────────┘ 
````                           

Tune lambda using cross-validation:

````julia
	 julia> lambdas, rmserrors = @curve λ logspace(-3,1,100) begin
               ridge.lambda = λ
               mean(cv(ridgeM, train, n_folds=9, verbosity=0))
           end;
````

Set lambda to the optimal value and do final train:

````julia
    julia> ridge.lambda = lambdas[indmin(rmserrors)]
    2.25701971963392

    julia> fit!(ridgeM, train)
    SupervisedMachine{RidgeRegressor}@...817

    julia> err(ridgeM, test)
    0.13403786668784523
````
