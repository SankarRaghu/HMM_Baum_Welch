using Random
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Plots
plotlyjs()
using LaTeXStrings
using DelimitedFiles

function readData(filename::String)
  data = readdlm(filename, ',')
  return Matrix{Float64}(data[2:end, 2:5])
end

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

function dataIndices(scaled_states::Matrix{Float64}, possible_states::Matrix{Float64})
  nT = size(scaled_states, 1)
  indices = zeros(Int64, nT)
  @inbounds for i in 1:nT
    pos = scaled_states[i, :]
    _, indices[i] = findmin(point -> norm(point .- pos, 2), eachrow(possible_states))
  end
  return indices
end

function gaussianPDF(x::Vector{Float64}, μ::Vector{Float64}, Σ::Matrix{Float64}, d::Int64)
  inv_Σ = inv(Σ)
  det_Σ = det(Σ)
  centered_μ = @. (x - μ)
  return (2 * π)^(-d / 2) * (abs(det_Σ))^(-1 / 2) *
         exp(-(1 / 2) * centered_μ' * inv_Σ * centered_μ)
end

function gaussianMixtureModel(obs::Vector{Float64}, μs::Matrix{Float64}, Σs::Array{Float64,3},
                              ws::Vector{Float64})
  sum_prob = 0.0
  @inbounds for k in 1:m
    sum_prob += ws[k] * gaussianPDF(obs, μs[k, :], Σs[k, :, :], 3)
  end
  return sum_prob
end

function initEmissionProbability(possible_states::Matrix{Float64}, μs::Array{Float64,3},
                                 Σs::Array{Float64,4}, ws::Matrix{Float64}, nS::Int64, nO::Int64)
  B = zeros(Float64, nS, nO)

  @inbounds for i in 1:nS
    @inbounds for j in 1:nO
      B[i, j] = gaussianMixtureModel(possible_states[j, :], μs[i, :, :], Σs[i, :, :, :], ws[i, :])
    end
  end
  return B
end

function initProbabilities(possible_states::Matrix{Float64}, μs::Array{Float64,3},
                           Σs::Array{Float64,4}, ws::Matrix{Float64}, nS::Int64, nO::Int64)
  # Initializing the initial probabilities of the states
  πs = (1 / nS) * ones(Float64, nS)

  # Initializing the transition probability matrix
  A = (1 / nS) * ones(Float64, nS, nS)

  # Initializing the emission probability matrix
  B = initEmissionProbability(possible_states, μs, Σs, ws, nS, nO)

  return πs, A, B
end

function forwardPass(πs::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64},
                     pos::Vector{Int64}, nS::Int64, nT::Int64)
  c = zeros(Float64, nT)
  α = zeros(Float64, nS, nT)
  @inbounds for i in 1:nS
    α[i, 1] = πs[i] * B[i, pos[1]]
  end
  c[1] = 1 / sum(α[:, 1])
  α[:, 1] *= c[1]

  @inbounds for t in 2:nT
    @inbounds for i in 1:nS
      sum_α = 0.0
      @inbounds for j in 1:nS
        sum_α += α[j, t - 1] * A[j, i]
      end
      α[i, t] = B[i, pos[t]] * sum_α
    end
    c[t] = 1 / sum(α[:, t])
    α[:, t] *= c[t]
  end

  lhood = -sum(log.(c))

  return α, c, lhood
end

function backwardPass(A::Matrix{Float64}, B::Matrix{Float64}, c::Vector{Float64},
                      pos::Vector{Int64}, nS::Int64,
                      nT::Int64)
  β = ones(Float64, nS, nT)
  β[:, end] *= c[end]

  @inbounds for t in (nT - 1):-1:1
    @inbounds for i in 1:nS
      sum_β = 0.0
      @inbounds for j in 1:nS
        sum_β += β[j, t + 1] * A[i, j] * B[j, pos[t + 1]]
      end
      β[i, t] = sum_β
    end
    β[:, t] *= c[t]
  end

  return β
end

function computeGamma(O::Matrix{Float64}, α::Matrix{Float64}, β::Matrix{Float64},
                      μs::Array{Float64,3}, Σs::Array{Float64,4}, ws::Matrix{Float64}, m::Int64,
                      nS::Int64, nT::Int64)
  γ = zeros(Float64, m, nS, nT)

  @inbounds for t in 1:nT
    @inbounds for j in 1:nS
      # Computing the sum for the GMM densities
      sum_gmm = gaussianMixtureModel(O[t, :], μs[j, :, :], Σs[j, :, :, :], ws[j, :])
      sum_αβ = 0.0
      @inbounds for i in 1:nS
        sum_αβ += α[i, t] * β[i, t]
      end
      # Computing the γ values for each component of the GMM model
      @inbounds for k in 1:m
        γ[k, j, t] = (α[j, t] * β[j, t]) * ws[j, k] *
                     gaussianPDF(O[t, :], μs[j, k, :], Σs[j, k, :, :], 3) / (sum_αβ * sum_gmm)
      end
    end
  end

  return γ
end

function computeXi(α::Matrix{Float64}, β::Matrix{Float64}, A::Matrix{Float64},
                   B::Matrix{Float64}, pos::Vector{Int64}, nS::Int64, nT::Int64)
  ξ = zeros(Float64, nS, nS, nT - 1)
  @inbounds for t in 1:(nT - 1)
    sum_ξ = 0.0
    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        sum_ξ += α[i, t] * A[i, j] * β[j, t + 1] * B[j, pos[t + 1]]
      end
    end

    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        ξ[i, j, t] = α[i, t] * A[i, j] * β[j, t + 1] * B[j, pos[t + 1]] / sum_ξ
      end
    end
  end

  return ξ
end

function parameterUpdate(O::Matrix{Float64}, πs::Vector{Float64},
                         A::Matrix{Float64}, γ::Array{Float64,3}, ξ::Array{Float64,3},
                         μs::Array{Float64,3}, Σs::Array{Float64,4}, ws::Matrix{Float64},
                         nS::Int64, nO::Int64, nT::Int64)
  ws_updated = zeros(Float64, nS, m)
  μs_updated = zeros(Float64, nS, m, 3)
  Σs_updated = zeros(Float64, nS, m, 3, 3)
  πs_updated = zeros(Float64, nS)
  A_updated = zeros(Float64, nS, nS)
  B_updated = zeros(Float64, nS, nO)

  # Computing the new parameters for the GMM model
  @inbounds for j in 1:nS
    sum_ws = sum(γ[:, j, :])
    @inbounds for k in 1:m
      ws_updated[j, k] = sum(γ[k, j, :]) / sum_ws
    end
  end
  ws_updated /= sum(ws_updated)

  @inbounds for j in 1:nS
    @inbounds for k in 1:m
      sum_μs = sum(γ[k, j, :])
      μs_updated[j, k, :] .= sum([γ[k, j, t] * O[t, :] for t in 1:nT]) / sum_μs
    end
  end

  @inbounds for j in 1:nS
    @inbounds for k in 1:m
      sum_Σs = sum(γ[k, j, :])
      Σs_updated[j, k, :, :] .= sum([γ[k, j, t] * (O[t, :] .- μs_updated[j, k, :]) *
                                     (O[t, :] .- μs_updated[j, k, :])' for t in 1:nT]) / sum_Σs
    end
  end

  # Updating the probabilities after learning from the training data
  πs_updated = vec(sum(γ[:, :, 1]; dims=1))
  πs_updated /= sum(πs_updated)

  @inbounds for i in 1:nS
    @inbounds for j in 1:nS
      sum_ξ = 0.0
      sum_γ = 0.0
      @inbounds for t in 1:(nT - 1)
        sum_ξ += ξ[i, j, t]
        sum_γ += sum(γ[:, i, t])
      end
      A_updated[i, j] = sum_ξ / sum_γ
    end
  end
  A_updated ./= sum(A_updated; dims=2)

  B_updated = initEmissionProbability(possible_states, μs_updated, Σs_updated, ws_updated, nS, nO)
  B_updated ./= sum(B_updated; dims=2)

  return πs_updated, A_updated, B_updated
end

function baumWelchAlgorithm(O::Matrix{Float64}, pos::Vector{Int64}, πs::Vector{Float64},
                            A::Matrix{Float64}, B::Matrix{Float64}, μs::Array{Float64,3},
                            Σs::Array{Float64,4}, ws::Matrix{Float64}, nS::Int64, nO::Int64,
                            nT::Int64, max_iter::Int64=1000, TOL::Float64=1e-3)
  prev_log_likelihood = -Inf

  @inbounds for _ in 1:max_iter
    α, c, lhood = forwardPass(πs, A, B, pos, nS, nT)
    β = backwardPass(A, B, c, pos, nS, nT)

    γ = computeGamma(O, α, β, μs, Σs, ws, m, nS, nT)
    ξ = computeXi(α, β, A, B, pos, nS, nT)

    # Updating the parameters πs, A, and B
    πs_new, A_new, B_new = parameterUpdate(O, πs, A, γ, ξ, μs, Σs, ws, nS, nO, nT)

    log_likelihood = lhood

    # Checking for convergence using the log-likelihood
    if abs(log_likelihood - prev_log_likelihood) < TOL
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

function predictObservation(O_d::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, d::Int64)

  # Computing the likelihood of the prediction over all possible states
  log_likelihood = zeros(Float64, nO)
  @inbounds for state in 1:nO
    # Assembling the extended state space
    O_ext = cat(O_d, state; dims=1)

    # Computing the likelihood of making the observations
    α, c, lhood = forwardPass(πs, A, B, O_ext, nS, d + 1)

    log_likelihood[state] = lhood
  end

  # Compute the state where the likelihood is the maximum
  prediction = argmax(log_likelihood)

  return prediction
end

"""
MAIN PROGRAM
"""

rng = MersenneTwister(1000)

# Reading the training data and scaling the features
training_data = readData("AAPL_Train.csv")
scaled_training_states = scalingFeatures(training_data)

# Computing the possible states that can be reached
minfC = -0.1
maxfC = 0.1
minfH = 0.0
maxfH = 0.1
minfL = 0.0
maxfL = 0.1
nC = 50
nH = 10
nL = 10
possible_states = possibleStates(minfC, maxfC, minfH, maxfH, minfL, maxfL, nC, nH, nL)

# Parameters needed for modeling the GMM-HMM model
d = 10
nS = 4
nO = (nC + 1) * (nH + 1) * (nL + 1)
nTrain = size(scaled_training_states, 1)

# Computing the initial parameters for the GMM model using the k-Means clustering algorithm
rng = MersenneTwister(1000)
m = 5
μs_init = begin
  local μ_vals = zeros(Float64, nS, m, 3)
  @inbounds for i in 1:nS
    vals = vec(sample(rng, eachrow(scaled_training_states), m; replace=false))
    @inbounds for k in 1:m
      μ_vals[i, k, :] .= vals[k][:]
    end
  end
  μ_vals
end
Σs_init = begin
  local Σ_vals = zeros(Float64, nS, m, 3, 3)
  @inbounds for i in 1:nS
    @inbounds for k in 1:m
      Σ_vals[i, k, :, :] .= Matrix{Float64}(I, 3, 3)
    end
  end
  Σ_vals
end
ws_init = begin
  local w_vals = zeros(Float64, nS, m)
  @inbounds for i in 1:nS
    @inbounds for k in 1:m
      w_vals[i, k] = 1 / m
    end
  end
  w_vals
end

indices = dataIndices(scaled_training_states, possible_states)

# Learning the parameters of the GMM-HMM model using the Baum--Welch algorithm
πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)

# # Initializing the probabilities
# πs_init, A_init, B_init = initProbabilities(possible_states, μs_init, Σs_init, ws_init, nS, nO)
# positions = indices
# O = scaled_training_states

# # Training the HMM model
# πs_train, A_train, B_train = baumWelchAlgorithm(O, positions, πs_init, A_init, B_init, μs_init,
#                                                 Σs_init, ws_init, nS, nO, nTrain)
# πs_tot .+= πs_train
# A_tot .+= A_train
# B_tot .+= B_train

@inbounds for i in 1:(nTrain - d + 1)
  # Printing the iteration
  println("Training iteration :: $i")

  # Initializing the probabilities
  πs_init, A_init, B_init = initProbabilities(possible_states, μs_init, Σs_init, ws_init, nS, nO)
  positions = indices[i:(i + d - 1)]
  O = scaled_training_states[i:(i + d - 1), :]

  # Training the HMM model
  πs_train, A_train, B_train = baumWelchAlgorithm(O, positions, πs_init, A_init, B_init, μs_init,
                                                  Σs_init, ws_init, nS, nO, d)
  global πs_tot .+= πs_train
  global A_tot .+= A_train
  global B_tot .+= B_train
end

# Normalizing the trained parameters
πs_tot /= sum(πs_tot)
A_tot ./= sum(A_tot; dims=2)
B_tot ./= sum(B_tot; dims=2)

# Reading the test data and scaling the features
test_data = readData("AAPL_Test.csv")
scaled_states = scalingFeatures(test_data)
nTest = size(scaled_states, 1)

# Computing the indices of the scaled test data wrt. the possible states
append!(indices, dataIndices(scaled_states, possible_states))

# Prediction of future states based on the trained parameters
prediction_indices = zeros(Int64, nTest)
@inbounds for i in 0:(nTest - 1)
  O_d = indices[(nTrain - d + i):(nTrain + i)]
  prediction_indices[i + 1] = predictObservation(O_d, πs_tot, A_tot, B_tot, nS, nO, d + 1)
end
display(prediction_indices)