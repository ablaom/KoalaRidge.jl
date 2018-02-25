module KoalaRidge

# new:
export RidgeRegressor # eg, RidgeRegressor

# needed in this module:
import Koala: Regressor, BaseType, SupervisedMachine, Transformer
import Koala: params, keys_ordered_by_values
import DataFrames: AbstractDataFrame, DataFrame
import KoalaTransforms: OneHotEncoder, OneHotEncoderScheme
import KoalaTransforms: BoxCoxTransformer, BoxCoxTransformerScheme
import KoalaTransforms: UnivariateStandardizer, UnivariateBoxCoxTransformer
import MultivariateStats
import UnicodePlots

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: get_transformer_X, get_transformer_y, transform, inverse_transform


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
    boxcox_inputs::Bool # whether to apply Box-Cox transformations to the input patterns
    boxcox::Bool # whether to apply Box-Cox transformations to the
                 # target (preceding any standarization)
    standardize::Bool   # whether to standardize targets
    shift::Bool # whether to shift away from zero in Box-Cox transformations
    drop_last::Bool # for hot-encoding, which is always performed
end

# lazy keywork constructor
RidgeRegressor(; lambda=0.0, boxcox_inputs::Bool=false,
               boxcox::Bool=false, standardize::Bool=true,
               shift::Bool=true, drop_last::Bool=true) =
                   RidgeRegressor(lambda, boxcox_inputs, boxcox, standardize,
                                  shift, drop_last)

struct Transformer_X <: Transformer
    boxcox_inputs::Bool # whether to apply Box-Cox transformations to the input patterns
    shift::Bool # do we shift away from zero in Box-Cox transformations?
    drop_last::Bool # do we drop the last slot in the one-hot-encoding?
end

struct Transformer_y <: Transformer
    boxcox::Bool # do we apply Box-Cox transforms to target (before any standarization)?
    standardize::Bool # do we standardize targets?
    shift::Bool # do we shift away from zero in Box-Cox transformations?
end

struct Scheme_X <: BaseType
    boxcox::BoxCoxTransformerScheme
    hot::OneHotEncoderScheme
    features::Vector{Symbol}
    spawned_features::Vector{Symbol} # ie after one-hot encoding
end

struct Scheme_y <: BaseType
    boxcox::Tuple{Float64,Float64}
    standard::Tuple{Float64,Float64}
end

function fit(transformer::Transformer_X, X::AbstractDataFrame, parallel, verbosity)

    features = names(X)
    
    # check `X` has only string and real eltypes:
    eltypes_ok = true
    for ft in features
        T = eltype(X[ft])
        if !(T <: AbstractString || T <: Real)
            eltypes_ok = false
        end
    end
    eltypes_ok || error("Only AbstractString and Real eltypes allowed in DataFrame.")

    # fit Box-Cox transformation:
    if transformer.boxcox_inputs
        info("Computing input Box-Cox transformations.")
        boxcox_transformer = BoxCoxTransformer(shift=transformer.shift)
        boxcox = fit(boxcox_transformer, X, true, verbosity - 1)
        X = transform(boxcox_transformer, boxcox, X)
    else
        boxcox = BoxCoxTransformerScheme() # null scheme
    end

    info("Determining one-hot encodings for inputs.")
    hot_transformer = OneHotEncoder(drop_last=transformer.drop_last)
    hot =  fit(hot_transformer, X, true, verbosity - 1) 
    spawned_features = hot.spawned_features    

    return Scheme_X(boxcox, hot, features, spawned_features)

end

function transform(transformer::Transformer_X, scheme_X, X::AbstractDataFrame)
    issubset(Set(scheme_X.features), Set(names(X))) ||
        error("DataFrame feature incompatibility encountered.")
    X = X[scheme_X.features]
    if transformer.boxcox_inputs
        boxcox_transformer = BoxCoxTransformer(shift=transformer.shift)
        X = transform(boxcox_transformer, scheme_X.boxcox, X)
    end
    hot_transformer = OneHotEncoder(drop_last=transformer.drop_last)
    X = transform(hot_transformer, scheme_X.hot, X)
    return convert(Array{Float64}, X)
end

function fit(transformer::Transformer_y, y, parallel, verbosity)

    if transformer.boxcox
        info("Computing Box-Cox transformations for target.")
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        boxcox = fit(boxcox_transformer, y, true, verbosity - 1)
        y = transform(boxcox_transformer, boxcox, y)
    else
        boxcox = (0.0, 0.0) # null scheme
    end
    if transformer.standardize
        info("Computing target standardization.")
        standard_transformer = UnivariateStandardizer()
        standard = fit(standard_transformer, y, true, verbosity - 1)
    else
        standard = (0.0, 1.0) # null scheme
    end
    return Scheme_y(boxcox, standard)
end 

function transform(transformer::Transformer_y, scheme_y, y)
    if transformer.boxcox
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        y = transform(boxcox_transformer, scheme_y.boxcox, y)
    end
    if transformer.standardize
        standard_transformer = UnivariateStandardizer()
        y = transform(standard_transformer, scheme_y.standard, y)
    end 
    return y
end 

function inverse_transform(transformer::Transformer_y, scheme_y, y)
    if transformer.standardize
        standard_transformer = UnivariateStandardizer()
        y = inverse_transform(standard_transformer, scheme_y.standard, y)
    end
    if transformer.boxcox
        boxcox_transformer = UnivariateBoxCoxTransformer(shift=transformer.shift)
        y = inverse_transform(boxcox_transformer, scheme_y.boxcox, y)
    end
    return y
end

get_transformer_X(model::RidgeRegressor) =
    Transformer_X(model.boxcox_inputs, model.shift, model.drop_last)
get_transformer_y(model::RidgeRegressor) =
    Transformer_y(model.boxcox, model.standardize, model.shift)

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






