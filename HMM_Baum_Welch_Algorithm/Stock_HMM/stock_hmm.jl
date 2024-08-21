using Pkg

# Installing the necessary libraries
Pkg.add(["Random", "Statistics", "StatsBase", "Distributions", "LinearAlgebra", "Plots", "LaTeXStrings", "DelimitedFiles", "ProgressBars", "MarketData", "Dates"])

using Random
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Plots
pgfplotsx()
using LaTeXStrings
using DelimitedFiles
using ProgressBars
using MarketData
using Dates

# Read stock market data through webscraping
function readData(symbol::String, start_date::Tuple{Int64,Int64,Int64},
                  end_date::Tuple{Int64,Int64,Int64}, intv::String)
  start_time = DateTime(start_date[1], start_date[2], start_date[3])
  end_time = DateTime(end_date[1], end_date[2], end_date[3])
  data = yahoo(symbol, YahooOpt(; period1=start_time, period2=end_time, interval=intv))
  return values(data["Open", "High", "Low", "Close"])
end

# Scaling the features from 4D state space to 3D state space
function scalingFeatures(data::Matrix{Float64})
  open = data[:, 1]
  high = data[:, 2]
  low = data[:, 3]
  close = data[:, 4]
  fC = @. (close - open) / open
  fH = @. (high - open) / open
  fL = @. (open - low) / open
  return cat(fC, fH, fL; dims=2)
end

# Generating the mesh of possible states that can be attained by the model
function possibleStates(minfC::Float64, maxfC::Float64, minfH::Float64, maxfH::Float64,
                        minfL::Float64, maxfL::Float64, nC::Int64, nH::Int64, nL::Int64)
  ΔfC = (maxfC - minfC) / nC
  ΔfH = (maxfH - minfH) / nH
  ΔfL = (maxfL - minfL) / nL
  possible_states = zeros(Float64, (nC + 1) * (nH + 1) * (nL + 1), 3)
  counter = 1
  @inbounds for i in 0:nC
    @inbounds for j in 0:nH
      @inbounds for k in 0:nL
        possible_states[counter, :] .= [minfC + i * ΔfC, minfH + j * ΔfH, minfL + k * ΔfL]
        counter += 1
      end
    end
  end
  return possible_states
end

# Projecting the actual stock data observations onto the generated mesh
function dataIndices(scaled_states::Matrix{Float64}, possible_states::Matrix{Float64})
  nT = size(scaled_states, 1)
  indices = zeros(Int64, nT)
  @inbounds for i in 1:nT
    pos = scaled_states[i, :]
    _, indices[i] = findmin(point -> norm(point .- pos, 2), eachrow(possible_states))
  end
  return indices
end

# Initializing the paratemeter set θ = (π, A, B) for training the HMM model
function initProbabilities(nS::Int64, nO::Int64, rng::MersenneTwister)
  # Initializing the initial probabilities of the states
  πs = (1 / nS) * ones(Float64, nS)

  # Initializing the transition probability matrix
  A = (1 / nS) * ones(Float64, nS, nS)

  # Initializing the emission probability matrix
  B = (1 / nO) * ones(Float64, nS, nO) .-
      (1 / nO) * Matrix{Float64}(rand(rng, Dirichlet(ones(nO)), nS)')

  return πs, A, B
end

function forwardPass(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64},
                     nS::Int64, nT::Int64)
  c = zeros(Float64, nT)
  α = zeros(Float64, nS, nT)
  @inbounds for i in 1:nS
    α[i, 1] = πs[i] * B[i, O[1]]
  end
  c[1] = 1 / sum(α[:, 1])
  α[:, 1] *= c[1]

  @inbounds for t in 2:nT
    @inbounds for i in 1:nS
      sum_α = 0.0
      @inbounds for j in 1:nS
        sum_α += α[j, t - 1] * A[j, i]
      end
      α[i, t] = B[i, O[t]] * sum_α
    end
    c[t] = 1 / sum(α[:, t])
    α[:, t] *= c[t]
  end

  return α, c
end

function backwardPass(O::Vector{Int64}, A::Matrix{Float64},
                      B::Matrix{Float64}, c::Vector{Float64}, nS::Int64, nT::Int64)
  β = ones(Float64, nS, nT)
  β[:, end] *= c[end]

  @inbounds for t in (nT - 1):-1:1
    @inbounds for i in 1:nS
      sum_β = 0.0
      @inbounds for j in 1:nS
        sum_β += β[j, t + 1] * A[i, j] * B[j, O[t + 1]]
      end
      β[i, t] = sum_β
    end
    β[:, t] *= c[t]
  end

  return β
end

function computeGamma(α::Matrix{Float64}, β::Matrix{Float64}, nS::Int64, nT::Int64)
  γ = zeros(Float64, nS, nT)

  @inbounds for t in 1:nT
    sum_αβ = sum(α[:, t] .* β[:, t])
    @inbounds for j in 1:nS
      γ[j, t] = (α[j, t] * β[j, t]) / sum_αβ
    end
  end

  return γ
end

function computeXi(O::Vector{Int64}, α::Matrix{Float64}, β::Matrix{Float64}, A::Matrix{Float64},
                   B::Matrix{Float64}, nS::Int64, nT::Int64)
  ξ = zeros(Float64, nS, nS, nT - 1)
  @inbounds for t in 1:(nT - 1)
    sum_ξ = 0.0
    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        sum_ξ += α[i, t] * A[i, j] * β[j, t + 1] * B[j, O[t + 1]]
      end
    end

    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        ξ[i, j, t] = α[i, t] * A[i, j] * β[j, t + 1] * B[j, O[t + 1]] / sum_ξ
      end
    end
  end

  return ξ
end

function parameterUpdate(O::Vector{Int64}, γ::Matrix{Float64}, ξ::Array{Float64,3}, nS::Int64,
                         nO::Int64, nT::Int64)
  πs_updated = zeros(Float64, nS)
  A_updated = zeros(Float64, nS, nS)
  B_updated = zeros(Float64, nS, nO)

  # Updating the probabilities after learning from the training data
  πs_updated = γ[:, 1]
  πs_updated /= sum(πs_updated)

  @inbounds for i in 1:nS
    @inbounds for j in 1:nS
      sum_ξ = 0.0
      sum_γ = 0.0
      @inbounds for t in 1:(nT - 1)
        sum_ξ += ξ[i, j, t]
        sum_γ += sum(γ[i, t])
      end
      A_updated[i, j] = sum_ξ / sum_γ
    end
  end
  A_updated ./= sum(A_updated; dims=2)

  @inbounds for j in 1:nS
    @inbounds for k in 1:nO
      numerator = 0.0
      denominator = 0.0
      @inbounds for t in 1:nT
        if O[t] == k
          numerator += γ[j, t]
        end
        denominator += γ[j, t]
      end
      B_updated[j, k] = numerator / denominator
    end
  end
  B_updated ./= sum(B_updated; dims=2)

  return πs_updated, A_updated, B_updated
end

function baumWelchAlgorithm(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, nT::Int64,
                            max_iter::Int64=10000, TOL::Float64=1e-10)
  prev_log_likelihood = -Inf

  @inbounds for _ in 1:max_iter
    α, c = forwardPass(O, πs, A, B, nS, nT)
    β = backwardPass(O, A, B, c, nS, nT)

    γ = computeGamma(α, β, nS, nT)
    ξ = computeXi(O, α, β, A, B, nS, nT)

    # Updating the parameters πs, A, and B
    πs_new, A_new, B_new = parameterUpdate(O, γ, ξ, nS, nO, nT)

    log_likelihood = -sum(log.(c))

    if abs(prev_log_likelihood - log_likelihood) < TOL
      break
    end
    printstyled("\t Log likelihood :: $(round(log_likelihood, sigdigits=10)) \n"; color=:blue)

    prev_log_likelihood = log_likelihood
    πs .= πs_new
    A .= A_new
    B .= B_new
  end

  return πs, A, B
end

function viterbiAlgorithm(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                          B::Matrix{Float64}, nS::Int64, nT::Int64)
  log_P = fill(-Inf, (nS, nT))
  P_p = zeros(Int64, nS, nT)
  @inbounds for i in 1:nS
    log_P[i, 1] = log(πs[i]) + log(B[i, O[1]])
  end

  @inbounds for t in 2:nT
    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        log_prob = log_P[j, t - 1] + log(A[j, i]) + log(B[i, O[t]])
        if log_prob > log_P[i, t]
          log_P[i, t] = log_prob
          P_p[i, t] = j
        end
      end
    end
  end

  path = zeros(Int64, nT)
  path[end] = argmax(log_P[:, end])

  @inbounds for t in (nT - 1):-1:1
    path[t] = P_p[path[t + 1], t + 1]
  end

  return path
end

# Generate future hidden states based on the trained HMM model
function genFutureStates(cS::Int64, A::Matrix{Float64}, nS::Int64, nT::Int64, rng::MersenneTwister)
  future_states = Vector{Int64}(undef, 0)
  @inbounds for _ in 1:nT
    w = ProbabilityWeights(A[cS, :])
    cS_new = sample(rng, collect(1:nS), w)
    append!(future_states, cS_new)
    cS = cS_new
  end
  return future_states
end

# Generate future observations based on the trained HMM model
function genFutureObs(fS::Vector{Int64}, B::Matrix{Float64}, nO::Int64, rng::MersenneTwister)
  future_observations = Vector{Int64}(undef, 0)
  @inbounds for s in fS
    w = ProbabilityWeights(B[s, :])
    observation = sample(rng, collect(1:nO), w)
    append!(future_observations, observation)
  end
  return future_observations
end

function predictFutureObs(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                          B::Matrix{Float64}, nS::Int64, nO::Int64, nT::Int64)
  likelihood = fill(-Inf, nO)
  @inbounds for k in 1:nO
    O_ext = cat(O, k; dims=1)

    α, c = forwardPass(O_ext, πs, A, B, nS, nT + 1)

    likelihood[k] = log(sum(α[:, end]))
  end
  prediction = argmax(likelihood)
  return prediction
end

"""
MAIN PROGRAM
"""

# Reading the training data and scaling the features
# symbol = "AAPL"
# symbol = "HYMTF"
# symbol = "BTC-USD"
symbol = "SMSN.IL"
training_data = readData(symbol, (2023, 1, 1), (2024, 5, 1), "1d")
test_data = readData(symbol, (2024, 5, 2), (2024, 8, 1), "1d")
scaled_train_states = scalingFeatures(training_data)
scaled_test_states = scalingFeatures(test_data)

# Computing the possible states that can be reached
minfC = minimum(scaled_train_states[:, 1])
maxfC = maximum(scaled_train_states[:, 1])
minfH = minimum(scaled_train_states[:, 2])
maxfH = maximum(scaled_train_states[:, 2])
minfL = minimum(scaled_train_states[:, 3])
maxfL = maximum(scaled_train_states[:, 3])
nC = 50
nH = 20
nL = 20
nO = (nC + 1) * (nH + 1) * (nL + 1)
possible_states = possibleStates(minfC, maxfC, minfH, maxfH, minfL, maxfL, nC, nH, nL)

# Parameters needed for modeling the GMM-HMM model
rng = MersenneTwister(1000)
nS = 3
nTrain = size(scaled_train_states, 1)
nTest = size(scaled_test_states, 1)

πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)

# Initializing the probabilities
πs_init, A_init, B_init = initProbabilities(nS, nO, rng)

# Training the HMM model
indices_train = dataIndices(scaled_train_states, possible_states)
πs_tot, A_tot, B_tot = baumWelchAlgorithm(indices_train, πs_init, A_init, B_init, nS, nO, nTrain)

indices_test = dataIndices(scaled_test_states, possible_states)

# Predicting the optimal states for the trained HMM model
optimal_states = viterbiAlgorithm(indices_train, πs_tot, A_tot, B_tot, nS, nTrain)

# Running Monte Carlo simulations to obtain an estimate for the close price
nMC = 10
future_observations = zeros(Int64, nTest, nMC)
future_close_vals = zeros(Float64, nTest, nMC)
@inbounds for k in 1:nMC
  println("Monte Carlo iteration :: $k")
  current_state = optimal_states[end]
  future_states = genFutureStates(current_state, A_tot, nS, nTest, rng)
  future_observations[:, k] .= genFutureObs(future_states, B_tot, nO, rng)

  @inbounds for i in 1:nTest
    open_val = test_data[i, 1]
    fC = possible_states[future_observations[i, k], 1]
    future_close_vals[i, k] = open_val * fC + open_val
  end
end

# Computing the mean and the variance of the predictions
mean_future_obs = mean(future_close_vals; dims=2)
var_future_obs = var(future_close_vals; dims=2)
error_vals = begin
  local Cα = 1.96
  Cα * sqrt.(var_future_obs / nMC)
end

# Computing the mean absolute prediction error (MAPE)
rel_vals = @. abs(mean_future_obs - test_data[:, 4]) / abs(test_data[:, 4]) * 100
MAPE = mean(rel_vals)
println("Mean Absolute Percentage Error (MAPE) :: ", round(MAPE; sigdigits=6))

# Plotting the stock predictions along with actual test data
default(; size=(1200, 1000), legendfontsize=24, guidefontsize=22, tickfontsize=20, titlefontsize=26,
        minorgrid=true, linewidth=1.5, framestyle=:box)
plot(1:nTest, mean_future_obs; ribbon=error_vals, label="Future predictions", linecolor="blue",
     markershape=:circle, markercolor="blue", markersize=4)
plot!(1:nTest, test_data[:, 4]; label="Actual stock price", linecolor="red", markershape=:circle,
      markercolor="red", markersize=4)
plot!(; legend_position=:topleft)
title!("$(symbol) stock actual vs. prediction")
xlabel!("Time (in days)")
ylabel!("Stock price (in \$)")
savefig("$(symbol)_predictions.pdf")