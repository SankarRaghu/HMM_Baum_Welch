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

function meshStates(minfC::Float64, maxfC::Float64, minfH::Float64, maxfH::Float64,
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

function dataPositions(scaled_states::Matrix{Float64}, possible_states::Matrix{Float64})
  nT = size(scaled_states, 1)
  indices = zeros(Int64, nT)
  @inbounds for i in 1:nT
    pos = scaled_states[i, :]
    _, indices[i] = findmin(point -> norm(point .- pos, 2), eachrow(possible_states))
  end
  return indices
end

function initProbabilities(nS::Int64, nO::Int64, rng::MersenneTwister)
  # Initializing the initial probabilities of the states
  πs = (1 / nS) * ones(Float64, nS)
  # πs = rand(rng, Float64, nS)
  # πs /= sum(πs)

  # Initializing the transition probability matrix
  A = (1 / nS) * ones(Float64, nS, nS)
  # A = (1 / nS) * rand(Float64, nS, nS)
  # A ./= sum(A; dims=2)

  # Initializing the emission probability matrix
  B = rand(rng, Float64, nS, nO)
  B ./= sum(B; dims=2)

  return πs, A, B
end

function forwardPass(πs::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, O::Vector{Int64},
                     nS::Int64, nT::Int64)
  c = zeros(Float64, nT)
  α = zeros(Float64, nS, nT)
  @inbounds for i in 1:nS
    α[i, 1] = πs[i] * B[i, O[1]]
  end
  v = sum(α[:, 1])
  c[1] = 1 / v
  α[:, 1] *= c[1]

  @inbounds for t in 2:nT
    @inbounds for i in 1:nS
      sum_α = 0.0
      @inbounds for j in 1:nS
        sum_α += α[j, t - 1] * A[j, i]
      end
      α[i, t] = B[i, O[t]] * sum_α
    end
    v = sum(α[:, t])
    c[t] = 1 / v
    α[:, t] *= c[t]
  end

  log_likelihood = -sum(log.(c))

  return α, c, log_likelihood
end

function backwardPass(A::Matrix{Float64}, B::Matrix{Float64}, c::Vector{Float64},
                      O::Vector{Int64}, nS::Int64, nT::Int64)
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

function parameterUpdate(O::Vector{Int64}, A::Matrix{Float64}, B::Matrix{Float64},
                         α::Matrix{Float64}, β::Matrix{Float64}, c::Vector{Float64}, nS::Int64,
                         nO::Int64, nT::Int64)
  πs_updated = zeros(Float64, nS)
  A_updated = zeros(Float64, nS, nS)
  B_updated = zeros(Float64, nS, nO)

  # Updating the prior probability vector πs
  πs_updated .= α[:, 1] .* β[:, 1] / sum(α[:, 1] .* β[:, 1])

  # Updating the transition probability matrix A
  @inbounds for i in 1:nS
    @inbounds for j in 1:nS
      numerator = 0.0
      @inbounds for t in 1:(nT - 1)
        numerator += α[i, t] * A[i, j] * B[j, O[t + 1]] * β[j, t + 1]
      end

      denominator = 0.0
      @inbounds for t in 1:(nT - 1)
        denominator += α[i, t] * β[i, t] / c[t]
      end

      A_updated[i, j] = numerator / denominator
    end
  end

  # Updating the emission probability matrix B
  @inbounds for j in 1:nS
    denominator = sum([α[j, t] * β[j, t] / c[t] for t in 1:nT])
    @inbounds for k in 1:nO
      numerator = sum([α[j, t] * β[j, t] / c[t] for t in 1:nT if O[t] == k])
      B_updated[j, k] = numerator / denominator
    end
  end

  return πs_updated, A_updated, B_updated
end

function baumWelchAlgorithm(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, nT::Int64,
                            max_iter::Int64=10000, TOL::Float64=1e-10)
  prev_log_likelihood = -Inf

  @inbounds for _ in 1:max_iter
    α, c, log_likelihood = forwardPass(πs, A, B, O, nS, nT)
    β = backwardPass(A, B, c, O, nS, nT)

    # Updating the parameters πs, A, and B
    πs_new, A_new, B_new = parameterUpdate(O, A, B, α, β, c, nS, nO, nT)

    πs .= πs_new
    A .= A_new
    B .= B_new

    log_likelihood = -sum(log.(c))

    # Checking for convergence using the log-likelihood
    err = abs(log_likelihood - prev_log_likelihood)
    if err < TOL
      break
    end
    printstyled("\t Log likelihood :: $(round(log_likelihood, sigdigits=10)) \n"; color=:blue)

    prev_log_likelihood = log_likelihood
  end

  return πs, A, B
end

function predictObservation(O_d::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, d::Int64)

  # Computing the likelihood of the prediction over all possible states
  log_likelihood = fill(-Inf, nO)

  @inbounds for state in 1:nO
    # Assembling the extended state space
    O_ext = cat(O_d, state; dims=1)

    # Computing the likelihood of making the observations
    α, c, lhood = forwardPass(πs, A, B, O_ext, nS, d + 1)

    log_likelihood[state] = lhood
  end
  println(log_likelihood)

  prediction = argmax(log_likelihood)

  return prediction
end

"""
MAIN PROGRAM
"""

# Reading the training data and scaling the features
training_data = readData("AAPL_Train.csv")
scaled_training_states = scalingFeatures(training_data)

# Computing the mesh of states
minfC = -0.1
maxfC = 0.1
minfH = 0.0
maxfH = 0.1
minfL = 0.0
maxfL = 0.1
nC = 50
nH = 10
nL = 10
mesh_states = meshStates(minfC, maxfC, minfH, maxfH, minfL, maxfL, nC, nH, nL)

# Computing the possible set of observations based on the training data
indices_training = dataPositions(scaled_training_states, mesh_states)

# Learning the parameters of the HMM model using the Baum--Welch algorithm
d = 10
nS = 4
nO = (nC + 1) * (nH + 1) * (nL + 1)
nTrain = size(scaled_training_states, 1)
πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)

@inbounds for i in 1:(nTrain - d + 1)
  println("Training iteration :: $i")
  rng = MersenneTwister(1000)

  # Initializing the probabilities
  πs_init, A_init, B_init = initProbabilities(nS, nO, rng)
  O = indices_training[i:(i + d - 1)]

  # Training the HMM model
  πs_train, A_train, B_train = baumWelchAlgorithm(O, πs_init, A_init, B_init, nS, nO, d)

  πs_tot .+= πs_train
  A_tot .+= A_train
  B_tot .+= B_train
end

πs_tot /= sum(πs_tot)
A_tot ./= sum(A_tot; dims=2)
B_tot ./= sum(B_tot; dims=2)

# Reading the test data and scaling the features
test_data = readData("AAPL_Test.csv")
scaled_test_states = scalingFeatures(test_data)
nTest = size(scaled_test_states, 1)

# Computing the indices of the scaled test data wrt. the possible states
indices_total = cat(indices_training, dataPositions(scaled_test_states, mesh_states); dims=1)

open("actual_positions.csv", "w") do io
  return writedlm(io, dataPositions(scaled_test_states, mesh_states), ',')
end

# Prediction of future states based on the trained parameters
prediction_indices = zeros(Int64, nTest)
@inbounds for i in 0:(nTest - 1)
  O_d = indices_total[(nTrain - d + 1 + i):(nTrain + i)]
  prediction_indices[i + 1] = predictObservation(O_d, πs_tot, A_tot, B_tot, nS, nO, d)
end

open("prediction_positions.csv", "w") do io
  return writedlm(io, prediction_indices, ',')
end