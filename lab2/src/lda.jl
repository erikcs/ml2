#using Distributions

function ldac2docterm(dictfile::AbstractString, documentfile::AbstractString)
    # Read a document in LDA-C format and return a document-word matrix
    # with corresponding dictionary
    dictionary = readdlm(dictfile)
    lines = readlines(documentfile)

    n_words = length(dictionary)
    n_topics = length(lines)

    # should really be a sparse matrix
    document_matrix = zeros(Int64, n_topics, n_words)

    for line in lines
        splt = split(line, " ")
        docnr = parse(Int64, splt[1])

        for entry in splt[2:end]
            word, count =  split(entry, ":")
            document_matrix[docnr, parse(Int64, word) + 1] = parse(Int64, count)
        end
    end

    document_matrix, dictionary
end

function lda(document_matrix::AbstractArray,
            alpha::Float64, beta::Float64, T::Int,
            n_iter=100, verbose=true, seed=1234)

    # Basic implementation of LDA with (collapsed) Gibbs sampling
    # Following the setup (same notation) from
    # Steyvers, Mark, and Tom Griffiths. "Probabilistic topic models."
    # Handbook of latent semantic analysis 427.7 (2007): 424-440.

    # Returns the learned word/topic distribution
    # and the learned topic/document distribution
    srand(seed)
    D, W = size(document_matrix) # n_documents, n_vocabulary
    #=N = sum(document_matrix) # n_word_tokens, not used=#

    TA = Dict{Tuple{Int64, Int64}, Int64}() # Topic assignment, document by topic
    CWT = zeros(Int64, W, T) # n_times word w is assigned to topic j
    CDT = zeros(Int64, D, T) # n_times topic j is assigned to some word token in document d

    # Pre compute the word-doc indices
    doc_word_idx = []
    for doc in 1:D
        push!(doc_word_idx, find(document_matrix[doc,:]))
    end

    # Initialize
    for doc in 1:D # For all documents
            for (i, word) in enumerate(doc_word_idx[doc]) # For each word in that document
            # Assign a random topic to that word
            z = rand(1:T)
            CWT[word, z] += 1
            CDT[doc, z] += 1
            TA[doc, i] = z
        end
    end

    # Pre compute the CWT and CDT sums as well
    sumCWT = vec(sum(CWT, 1))
    sumCDT = vec(sum(CDT, 1))

    START = time()
    for iter in 1:n_iter
        if verbose
            if iter % 50 == 0
                println("iteration: $iter, elapsed time:$(
                @sprintf("%.2f", time() - START))s")
            end
        end
        for doc in 1:D
            for (i, word) in enumerate(doc_word_idx[doc])
                # Decrement
                z = TA[doc, i]
                CWT[word, z] -= 1
                CDT[doc, z] -= 1

                # Conditional probs
                # How much each topic "like" this word (smoothed by `beta`)
                left = (CWT[word, :] + beta) ./
                        (sumCWT + W * beta)
                # How much this word "like" the current topic (smoothed by `alpha`)
                right = (CDT[doc, :] + alpha ) ./
                        (sumCDT + T * alpha)
                pz = left .* right
                pz /= sum(pz)

                # Sample new topic
                # z = indmax(rand(Multinomial(1, pz / sum(pz))))
                # faster:
                z = searchsortedfirst(cumsum(pz), rand())

                # Update assignments
                CWT[word, z] += 1
                CDT[doc, z] += 1
                TA[doc, i] = z
            end
        end
    end

    # Probs
    phi = (CWT + beta) # p(w|z), word topic distribution
    phi ./= sum(CWT, 1)
    theta = (CDT + alpha) # p(z), topic document distribution
    theta ./= sum(theta, 2)

    phi, theta
end
