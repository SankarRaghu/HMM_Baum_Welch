using Random
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Plots
plotlyjs()
using LaTeXStrings
using DelimitedFiles

# Function definitions to scale data to log-scale
function eexp(x::Float64)
  if isnan(x)
    return 0
  else
    return exp(x)
  end
end

function eln(x::Float64)
  if x == 0
    return NaN
  elseif x > 0
    return log(x)
  else
    return error("Negative input error")
  end
end

function elnsum(elnx::Float64, elny::Float64)
  if isnan(elnx) || isnan(elny)
    if isnan(elnx)
      return elny
    else
      return elnx
    end
  else
    if elnx > elny
      return elnx + eln(1 + exp(elny - elnx))
    else
      return elny + eln(1 + exp(elnx - elny))
    end
  end
end

function elnproduct(elnx::Float64, elny::Float64)
  if isnan(elnx) || isnan(elny)
    return NaN
  else
    return elnx + elny
  end
end

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

function kMeansClustering(scaled_data::Matrix{Float64}, m::Int64, rng::MersenneTwister)
  # Initializing the means of the clusters
  μs = zeros(Float64, m, 3)
  μ = sample(rng, eachrow(scaled_data), m; replace=false)
  @inbounds for k in 1:m
    μs[k, :] .= vec(μ[k])
  end

  # Running the k-means clustering algorithm
  clusters = [Matrix{Float64}(undef, 0, 3) for _ in 1:m]
  converged = false
  while !converged
    # Clearing the previously computed cluster points
    clusters = [Matrix{Float64}(undef, 0, 3) for _ in 1:m]

    # Assigning each point to the cluster based on the distance
    @inbounds for point in eachrow(scaled_data)
      centroid_distances = [norm(point .- centroid) for centroid in eachrow(μs)]
      assign_cluster = argmin(centroid_distances)
      clusters[assign_cluster] = vcat(clusters[assign_cluster], point')
    end

    # Updating the centroids based on the points in the clusters
    new_μs = begin
      data_vals = zeros(Float64, m, 3)
      @inbounds for k in 1:m
        if isempty(clusters[k])
          data_vals[k, :] .= μs[k, :]
        else
          data_vals[k, :] .= vec(mean(clusters[k]; dims=1))
        end
      end
      data_vals
    end

    # Computing the error associated with the centroids
    converged = (μs == new_μs)
    μs .= new_μs
  end

  # Initializing the covariances of the clusters
  Σs = zeros(Float64, m, 3, 3)

  # Computing the covariance matrix associated with the clusters
  @inbounds for k in 1:m
    matsum = zeros(Float64, 3, 3)
    points = clusters[k]
    @inbounds for point in eachrow(points)
      matsum .+= (point .- μs[k, :]) * (point .- μs[k, :])'
    end
    Σs[k, :, :] .= matsum / (size(points, 1) - 1)
  end

  # Computing the weights associated with the cluster points
  n_total = size(scaled_data, 1)
  w = [size(clusters[k], 1) / n_total for k in 1:m]

  return μs, Σs, w
end

function gaussianPDF(x::Vector{Float64}, μ::Vector{Float64}, Σ::Matrix{Float64}, d::Int64)
  inv_Σ = inv(Σ)
  det_Σ = det(Σ)
  centered_μ = @. (x - μ)
  return (2 * π)^(-d / 2) * (det_Σ)^(-1 / 2) *
         exp(-(1 / 2) * centered_μ' * inv_Σ * centered_μ)
end

function gaussianMixtureModel(state::Vector{Float64}, μs::Matrix{Float64}, Σs::Array{Float64,3},
                              ws::Vector{Float64})
  sum_prob = 0.0
  @inbounds for k in 1:m
    sum_prob += ws[k] * gaussianPDF(state, μs[k, :], Σs[k, :, :], 3)
  end
  return sum_prob
end

function initProbabilities(pos_states::Matrix{Float64}, μs::Matrix{Float64},
                           Σs::Array{Float64,3}, ws::Vector{Float64}, nS::Int64, nO::Int64)
  # Initializing the initial probabilities of the states
  πs = ones(nS)
  πs /= sum(πs)

  # Initializing the transition probability matrix
  A = ones(nS, nS)
  A ./= sum(A; dims=2)

  # Initializing the emission probability matrix
  B = zeros(Float64, nS, nO)
  @inbounds for i in 1:nS
    @inbounds for j in 1:nO
      B[i, j] = gaussianMixtureModel(pos_states[j, :], μs, Σs, ws)
    end
    B[i, :] ./= sum(B[i, :])
  end

  return πs, A, B
end

function forwardPass(πs::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, O::Vector{Int64},
                     nS::Int64, nT::Int64)
  elnα = zeros(Float64, nS, nT)
  @inbounds for i in 1:nS
    elnα[i, 1] = elnproduct(eln(πs[i]), eln(B[i, O[1]]))
  end

  @inbounds for t in 2:nT
    @inbounds for j in 1:nS
      logα = NaN
      @inbounds for i in 1:nS
        logα = elnsum(logα, elnproduct(elnα[i, t - 1], eln(A[i, j])))
      end
      elnα[j, t] = elnproduct(logα, eln(B[j, O[t]]))
    end
  end

  return elnα
end

function backwardPass(A::Matrix{Float64}, B::Matrix{Float64}, O::Vector{Int64}, nS::Int64,
                      nT::Int64)
  elnβ = zeros(Float64, nS, nT)

  @inbounds for t in (nT - 1):-1:1
    @inbounds for i in 1:nS
      logβ = NaN
      @inbounds for j in 1:nS
        logβ = elnsum(logβ, elnproduct(eln(A[i, j]), elnproduct(B[j, O[t + 1]], elnβ[j, t + 1])))
      end
      elnβ[i, t] = logβ
    end
  end

  return elnβ
end

function computeGamma(elnα::Matrix{Float64}, elnβ::Matrix{Float64}, nS::Int64, nT::Int64)
  elnγ = zeros(Float64, nS, nT)
  @inbounds for t in 1:nT
    normalizer = NaN
    @inbounds for i in 1:nS
      elnγ[i, t] = elnproduct(elnα[i, t], elnβ[i, t])
      normalizer = elnsum(normalizer, elnγ[i, t])
    end
    @inbounds for i in 1:nS
      elnγ[i, t] = elnproduct(elnγ[i, t], -normalizer)
    end
  end
  return elnγ
end

function computeXi(O::Vector{Int64}, A::Matrix{Float64}, B::Matrix{Float64}, elnα::Matrix{Float64},
                   elnβ::Matrix{Float64}, nS::Int64, nT::Int64)
  elnξ = zeros(Float64, nS, nS, nT - 1)
  @inbounds for t in 1:(nT - 1)
    normalizer = NaN
    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        elnξ[i, j, t] = elnproduct(elnα[i, t],
                                   elnproduct(eln(A[i, j]),
                                              elnproduct(eln(B[j, O[t + 1]]), elnβ[j, t + 1])))
        normalizer = elnsum(normalizer, elnξ[i, j, t])
      end
    end
    @inbounds for i in 1:nS
      @inbounds for j in 1:nS
        elnξ[i, j, t] = elnproduct(elnξ[i, j, t], -normalizer)
      end
    end
  end
  return elnξ
end

function parameterUpdate(O::Vector{Int64}, elnγ::Matrix{Float64}, elnξ::Array{Float64,3}, nS::Int64,
                         nO::Int64, nT::Int64)
  πs_updated = zeros(Float64, nS)
  A_updated = zeros(Float64, nS, nS)
  B_updated = zeros(Float64, nS, nO)

  # Updating the prior probability vector πs
  @inbounds for i in 1:nS
    πs_updated[i] = eexp(elnγ[i, 1])
  end

  # Updating the transition probability matrix A
  @inbounds for i in 1:nS
    @inbounds for j in 1:nS
      numerator = NaN
      denominator = NaN
      @inbounds for t in 1:(nT - 1)
        numerator = elnsum(numerator, elnξ[i, j, t])
        denominator = elnsum(denominator, elnγ[i, t])
      end
      A_updated[i, j] = eexp(elnproduct(numerator, -denominator))
    end
  end

  # Updating the emission probability matrix B
  @inbounds for k in 1:nO
    @inbounds for j in 1:nS
      numerator = NaN
      denominator = NaN
      @inbounds for t in 1:nT
        if O[t] == k
          numerator = elnsum(numerator, elnγ[j, t])
        end
        denominator = elnsum(denominator, elnγ[j, t])
      end
      B_updated[j, k] = eexp(elnproduct(numerator, -denominator))
    end
  end

  return πs_updated, A_updated, B_updated
end

function baumWelchAlgorithm(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, nT::Int64,
                            max_iter::Int64=1000, TOL::Float64=1e-6)
  prev_log_likelihood = -Inf
  @inbounds for _ in 1:max_iter
    elnα = forwardPass(πs, A, B, O, nS, nT)
    elnβ = backwardPass(A, B, O, nS, nT)
    elnγ = computeGamma(elnα, elnβ, nS, nT)
    elnξ = computeXi(O, A, B, elnα, elnβ, nS, nT)

    # Updating the parameters πs, A, and B
    πs_new, A_new, B_new = parameterUpdate(O, elnγ, elnξ, nS, nO, nT)

    log_likelihood = log(sum(eexp.(elnα[:, end])))

    if abs(log_likelihood - prev_log_likelihood) < TOL
      break
    end
    printstyled("\t Log-likelihood :: $(round.(log_likelihood, sigdigits=10)) \n"; color=:blue)

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
  log_likelihood = fill(-Inf, nO)

  @inbounds for state in 1:nO
    # Assembling the extended state space
    O_ext = cat(O_d, state; dims=1)

    # Computing the likelihood of making the observations
    elnα = forwardPass(πs, A, B, O_ext, nS, d + 1)

    log_likelihood[state] = sum(eexp.(elnα[:, end]))
  end
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
m = 5
nS = 4
nO = (nC + 1) * (nH + 1) * (nL + 1)
nTrain = size(scaled_training_states, 1)

# Running the k-means clustering algorithm for learning the emission probabiltites
rng = MersenneTwister(10000)
μs_init, Σs_init, ws_init = kMeansClustering(scaled_training_states, m, rng)

πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)

πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)
@inbounds for i in 1:(nTrain - d + 1)
  println("Training iteration :: $i")

  # Initializing the probabilities
  πs_init, A_init, B_init = initProbabilities(mesh_states, μs_init, Σs_init, ws_init, nS, nO)
  O = indices_training[i:(i + d - 1)]

  # Training the HMM model
  πs_train, A_train, B_train = baumWelchAlgorithm(O, πs_init, A_init, B_init, nS, nO, d)
  global πs_tot .+= πs_train
  global A_tot .+= A_train
  global B_tot .+= B_train
end

πs_tot /= sum(πs_tot)
A_tot ./= sum(A_tot; dims=2)
B_tot ./= sum(B_tot; dims=2)

open("emission_prob.csv", "w") do io
  return writedlm(io, B_tot', ',')
end

# Reading the test data and scaling the features
test_data = readData("AAPL_Test.csv")
scaled_test_states = scalingFeatures(test_data)
nTest = size(scaled_test_states, 1)

# Computing the indices of the scaled test data wrt. the possible states
indices_total = cat(indices_training, dataPositions(scaled_test_states, mesh_states); dims=1)

# Prediction of future states based on the trained parameters
prediction_indices = zeros(Int64, nTest)
@inbounds for i in 0:(nTest - 1)
  O_d = indices_total[(nTrain - d + 1 + i):(nTrain + i)]
  prediction_indices[i + 1] = predictObservation(O_d, πs_tot, A_tot, B_tot, nS, nO, d)
end
display(prediction_indices)