module KoalaRidge

# new:
export RidgeRegressor # eg, RidgeRegressor

# needed in this module:
import Koala: Regressor, BaseType, keys_ordered_by_values, SupervisedMachine
import Koala: params
import DataFrames: AbstractDataFrame, DataFrame
import KoalaTransforms: BoxCoxScheme, HotEncodingScheme
import KoalaTransforms: UnivariateBoxCoxScheme, UnivariateStandardizationScheme
import MultivariateStats
import UnicodePlots

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: get_scheme_X, get_scheme_y, transform, inverse_transform


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

mutable struct Scheme_X <: BaseType
    boxcox::BoxCoxScheme
    hot::HotEncodingScheme
    features::Vector{Symbol}
    spawned_features::Vector{Symbol} # ie after one-hot encoding
end

# `showall` method for `ElasticNetRegressor` machines:
function Base.showall(stream::IO,
                      mach::SupervisedMachine{LinearPredictor, RidgeRegressor})
    show(stream, mach)
    println(stream)
    if isdefined(mach,:report) && :feature_importance_curve in keys(mach.report)
        features, importance = mach.report[:feature_importance_curve]
        plt = UnicodePlots.barplot(features, importance,
              title="Feature importance (coefs of linear predictor)")
    end
    dict = params(mach)
    report_items = sort(collect(keys(dict[:report])))
    dict[:report] = "Dict with keys: $report_items"
    dict[:Xt] = string(typeof(mach.Xt), " of shape ", size(mach.Xt))
    dict[:yt] = string(typeof(mach.yt), " of shape ", size(mach.yt))
    delete!(dict, :cache)
    showall(stream, dict)
    println(stream, "\nModel detail:")
    showall(stream, mach.model)
    if isdefined(mach,:report) && :feature_importance_curve in keys(mach.report)
        show(stream, plt)
    end
end
    
function get_scheme_X(model::RidgeRegressor, X::AbstractDataFrame,
                      train_rows, features) 
    
    X = X[train_rows, features]

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
    if model.boxcox_inputs
        info("Computing input Box-Cox transformations.")
        boxcox = BoxCoxScheme(X, shift=model.shift)
        X = transform(boxcox, X)
    else
        boxcox = BoxCoxScheme()
    end

    info("Determining one-hot encodings for inputs.")
    hot =  HotEncodingScheme(X, drop_last=model.drop_last)
    spawned_features = hot.spawned_features    

    return Scheme_X(boxcox, hot, features, spawned_features)

end

function transform(model::RidgeRegressor, scheme_X, X::AbstractDataFrame)
    issubset(Set(scheme_X.features), Set(names(X))) ||
        error("DataFrame feature incompatibility encountered.")
    X = X[scheme_X.features]
    if model.boxcox_inputs
        X = transform(scheme_X.boxcox, X)
    end
    X = transform(scheme_X.hot, X)
    return convert(Array{Float64}, X)
end

mutable struct Scheme_y <: BaseType
    boxcox::UnivariateBoxCoxScheme
    standard::UnivariateStandardizationScheme
end

function get_scheme_y(model::RidgeRegressor, y, test_rows)
    y = y[test_rows]
    if model.boxcox
        info("Computing Box-Cox transformations for target.")
        boxcox = UnivariateBoxCoxScheme(y, shift=model.shift)
        y = transform(boxcox, y)
    else
        boxcox = UnivariateBoxCoxScheme()
    end
    if model.standardize
        info("Computing target standardization.")
        standard = UnivariateStandardizationScheme(y)
    else
        standard = UnivariateStandardizationScheme()
    end
    return Scheme_y(boxcox, standard)
end 
                          
function transform(model::RidgeRegressor, scheme_y , y::Vector{T} where T <: Real)
    if model.boxcox
        y = transform(scheme_y.boxcox, y)
    end
    if model.standardize
        y = transform(scheme_y.standard, y)
    end 
    return y
end 

function inverse_transform(model::RidgeRegressor, scheme_y, yt::AbstractVector)
    y = inverse_transform(scheme_y.standard, yt)
    if model.boxcox
        return inverse_transform(scheme_y.boxcox, y)
    else
        return y
    end
end 

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






