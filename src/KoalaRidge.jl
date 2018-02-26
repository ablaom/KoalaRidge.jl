module KoalaRidge

# new:
export RidgeRegressor # eg, RidgeRegressor

# needed in this module:
import Koala: Regressor, BaseType, SupervisedMachine, Transformer
import Koala: params, keys_ordered_by_values
import DataFrames: AbstractDataFrame, DataFrame
import KoalaTransforms
import MultivariateStats
import UnicodePlots

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, transform, inverse_transform


## Model type definitions

struct LinearPredictor <: BaseType
    coefficients::Vector{Float64}
    bias::Float64
end

# Following returns a `DataFrame` with three columns:
#
# column name | description
# :-----------|:-------------------------------------------------
# `:index`    | index of a feature used to train `predictor`
# `:feature`  | corresponding feature label provided by `features`
# `:coef`     | coefficient for that feature in the predictor
#
# The rows are ordered by the absolute value of the coefficients. If
# `rgs` is unfitted, an error is returned.
function coef_info(predictor::LinearPredictor, features)
    coef_given_index = Dict{Int, Float64}()
    abs_coef_given_index = Dict{Int, Float64}()
    v = predictor.coefficients
    for k in eachindex(v)
        coef_given_index[k] = v[k]
        abs_coef_given_index[k] = abs(v[k])
    end
    df = DataFrame()
    df[:index] = reverse(keys_ordered_by_values(abs_coef_given_index))
    df[:feature] = map(df[:index]) do index
        features[index]
    end
    df[:coef] = map(df[:index]) do index
        coef_given_index[index]
    end
    return df
end

mutable struct RidgeRegressor <: Regressor{LinearPredictor}
    lambda::Float64
end

# lazy keywork constructor
RidgeRegressor(; lambda=0.0) = RidgeRegressor(lambda)

default_transformer_X(model::RidgeRegressor) =
    KoalaTransforms.TransformerForLinearModels_X()
default_transformer_y(model::RidgeRegressor) =
    KoalaTransforms.TransformerForLinearModels_y()

struct Cache <: BaseType
    X::Matrix{Float64}
    y::Vector{Float64}
    features::Vector{Symbol}
end

setup(model::RidgeRegressor, X, y, scheme_X, parallel, verbosity) =
    Cache(X, y, scheme_X.spawned_features)

function fit(model::RidgeRegressor,
             cache, add, parallel, verbosity; args...)
    
    weights = MultivariateStats.ridge(cache.X, cache.y, model.lambda)

    coefficients = weights[1:end-1]
    bias = weights[end]

    predictor = LinearPredictor(coefficients, bias)

    # report on the relative strength of each feature in the predictor:
    report = Dict{Symbol, Any}()
    cinfo = coef_info(predictor, cache.features) # a DataFrame object
    u = Symbol[]
    v = Float64[]
    for i in 1:size(cinfo, 1)
        feature, coef = (cinfo[i, :feature], cinfo[i, :coef])
        coef = floor(1000*coef)/1000
        if coef < 0
            label = string(feature, " (-)")
        else
            label = string(feature, " (+)")
        end
        push!(u, label)
        push!(v, abs(coef))
    end
    report[:feature_importance_curve] = u, v

    return predictor, report, cache

end
    
function predict(predictor::LinearPredictor, pattern::Vector{Float64})

    ret = predictor.bias
    for i in eachindex(predictor.coefficients)
        ret += predictor.coefficients[i]*pattern[i]
    end
    
    return ret
    
end

predict(model::RidgeRegressor, predictor::LinearPredictor,
        X::Matrix{Float64}, parallel, verbosity) =
            Float64[predict(predictor, X[i,:]) for i in 1:size(X,1)]

end # of module






