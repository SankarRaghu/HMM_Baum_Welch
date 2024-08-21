using Random
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Plots
plotlyjs()
using LaTeXStrings

function initProbabilities(nS::Int64, nO::Int64, rng::MersenneTwister)
  # Initializing the initial probabilities of the states
  πs = rand(rng, Float64, nS)
  πs /= sum(πs)

  # Initializing the transition probability matrix
  A = rand(rng, Float64, nS, nS)
  A ./= sum(A; dims=2)

  # Initializing the emission probability matrix
  B = rand(rng, Float64, nS, nO)
  B ./= sum(B; dims=2)

  return πs, A, B
end

function forwardPass(πs::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, O::Vector{Int64},
                     nS::Int64, nTrain::Int64)
  c = zeros(Float64, nTrain)
  α = zeros(Float64, nS, nTrain)
  @inbounds for i in 1:nS
    α[i, 1] = πs[i] * B[i, O[1]]
  end
  c[1] = 1 / sum(α[:, 1])
  α[:, 1] *= c[1]

  @inbounds for t in 2:nTrain
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

  log_likelihood = sum(log.(sum(α[:, nTrain])))

  return α, c, log_likelihood
end

function backwardPass(πs::Vector{Float64}, A::Matrix{Float64},
                      B::Matrix{Float64}, c::Vector{Float64}, O::Vector{Int64}, nS::Int64,
                      nTrain::Int64)
  β = ones(Float64, nS, nTrain)
  β[:, end] *= c[end]

  @inbounds for t in (nTrain - 1):-1:1
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

function computeGamma(α::Matrix{Float64}, β::Matrix{Float64},
                      nS::Int64, nO::Int64, nTrain::Int64)
  γ = zeros(Float64, nS, nTrain)
  @inbounds for t in 1:nTrain
    @inbounds for i in 1:nS
      sum_αβ = 0.0
      @inbounds for j in 1:nS
        sum_αβ += α[j, t] * β[j, t]
      end
      γ[i, t] = α[i, t] * β[i, t] / sum_αβ
    end
  end

  return γ
end

function computeXi(α::Matrix{Float64}, β::Matrix{Float64}, A::Matrix{Float64},
                   B::Matrix{Float64}, O::Vector{Int64}, nS::Int64, nTrain::Int64)
  ξ = zeros(Float64, nS, nS, nTrain - 1)
  @inbounds for t in 1:(nTrain - 1)
    sum_ξ = 0.0
    @inbounds for k in 1:nS
      @inbounds for w in 1:nS
        sum_ξ += α[k, t] * A[k, w] * β[w, t + 1] * B[w, O[t + 1]]
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

function parameterUpdate(O::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                         B::Matrix{Float64}, γ::Matrix{Float64}, ξ::Array{Float64,3},
                         nS::Int64, nO::Int64, nTrain::Int64)
  πs .= γ[:, 1]

  @inbounds for i in 1:nS
    @inbounds for j in 1:nS
      sum_ξ = 0.0
      sum_γ = 0.0
      @inbounds for t in 1:(nTrain - 1)
        sum_ξ = ξ[i, j, t]
        sum_γ = γ[i, t]
      end
      A[i, j] = sum_ξ / sum_γ
    end
  end

  @inbounds for i in 1:nS
    @inbounds for k in 1:nO
      B[i, k] = sum([γ[i, t] for t in 1:nTrain if O[t] == k]) / sum(γ[i, :])
    end
  end

  return πs, A, B
end

function baumWelchAlgorithm(O::Vector{Int64}, nS::Int64, nO::Int64,
                            nTrain::Int64, rng::MersenneTwister, max_iter::Int64=1000,
                            TOL::Float64=1e-10)
  # Initializing the probabilities
  πs, A, B = initProbabilities(nS, nO, rng)

  prev_log_likelihood = -Inf

  @inbounds for iter in 1:max_iter
    α, c, log_likelihood = forwardPass(πs, A, B, O, nS, nTrain)
    β = backwardPass(πs, A, B, c, O, nS, nTrain)

    γ = computeGamma(α, β, nS, nO, nTrain)
    ξ = computeXi(α, β, A, B, O, nS, nTrain)

    # Updating the parameters πs, A, and B
    πs, A, B = parameterUpdate(O, πs, A, B, γ, ξ, nS, nO, nTrain)

    # Checking for convergence using the log-likelihood
    if abs(log_likelihood - prev_log_likelihood) < TOL
      break
    end
    prev_log_likelihood = log_likelihood
  end

  return πs, A, B
end

function predictObservation(O_d::Vector{Int64}, πs::Vector{Float64}, A::Matrix{Float64},
                            B::Matrix{Float64}, nS::Int64, nO::Int64, d::Int64)

  # Computing the likelihood of the prediction over all possible states
  log_likelihood = fill(Inf, nO)
  @inbounds for state in 1:nO
    # Assembling the extended state space
    O_ext = cat(O_d, state; dims=1)

    # Computing the likelihood of making the observations
    α, _, lhood = forwardPass(πs, A, B, O_ext, nS, d + 1)
    log_likelihood[state] = lhood
  end

  # Compute the state where the likelihood is the maximum
  prediction = argmax(log_likelihood)

  return prediction
end

rng = MersenneTwister(1000)

nS = 5
nO = 20
nTrain = 200
nTest = 50
d = 20

rng = MersenneTwister(1000)
observations = sample(rng, 1:nO, nTrain + nTest)

# Learning the parameters of the HMM model for different lag periods
πs_tot = zeros(Float64, nS)
A_tot = zeros(Float64, nS, nS)
B_tot = zeros(Float64, nS, nO)
@inbounds for i in 0:(nTrain - d)
  πs, A, B = baumWelchAlgorithm(observations[(i + 1):(i + d)], nS, nO, d, rng)
  πs_tot .+= πs / (nTrain - d)
  A_tot .+= A / (nTrain - d)
  B_tot .+= B / (nTrain - d)
end

predictions = zeros(Int64, nTest)
@inbounds for i in 0:(nTest - 1)
  local O_d = observations[(nTrain - d + 1 + i):(nTrain + i)]
  predictions[i + 1] = predictObservation(O_d, πs_tot, A_tot, B_tot, nS, nO, d)
end

default(; size=(1000, 1000))
scatter(1:(nTrain + nTest), observations; markercolor="red")
scatter!((nTrain + 1):(nTrain + nTest), predictions; markercolor="blue", markersize=4,
         markeralpha=0.25)
gui()

readline()