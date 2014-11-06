class NeuralNet
  attr_reader :shape, :weights, :outputs

  DEFAULT_TRAINING_OPTIONS = {
    max_iterations:   20_000,
    learning_rate:    0.3,
    momentum:         0.1,
    error_threshold:  0.005
  }

  def initialize(shape)
    @shape = shape
    @output_layer = @shape.length - 1
    set_random_weight_values
  end

  def train data, opts = {}
    opts = DEFAULT_TRAINING_OPTIONS.merge(opts)
    error_threshold, log_every = opts[:error_threshold], opts[:log_every]
    iteration = 0
    error = nil

    set_weight_changes_to_zeros

    while iteration < opts[:max_iterations]
      iteration += 1

      error = train_on_batch(data, opts[:learning_rate], opts[:momentum])
      
      if log_every && (iteration % log_every == 0)
        puts "[#{iteration}] error: #{error.round(5)}"
      end

      break if error_threshold && (error < error_threshold)
    end

    {error: error.round(5), iterations: iteration, below_error_threshold: (error < error_threshold)}
  end

  def run input
    # Input to this method represents the output of the first layer (i.e., the input layer)
    @outputs = [input]

    # Now calculate output of neurons in subsequent layers:
    1.upto(@output_layer).each do |layer|
      source_layer = layer - 1 # i.e, the layer that is feeding into this one
      source_outputs = @outputs[source_layer]

      @outputs[layer] = @weights[layer].map do |neuron_weights|
        # inputs to this neuron are the neuron outputs from the source layer times weights
        inputs = neuron_weights.map.with_index do |weight, i| 
          source_output = source_outputs[i] || 1 # if no output, this is the bias neuron
          weight * source_output
        end

        sum_of_inputs = inputs.reduce(:+)
        # the activated output of this neuron (using sigmoid activation function)
        sigmoid sum_of_inputs
      end
    end

    # Outputs of neurons in the last layer is the final result
    @outputs[@output_layer]
  end

  private

    def train_on_batch data, learning_rate, momentum
      total_error = 0

      set_gradients_to_zeroes

      data.each do |(input, ideal_output)|
        run input

        training_error = calculate_training_error ideal_output
        calculate_deltas training_error
        calculate_gradients
        
        total_error += mean_squared_error training_error
      end

      # update weights using gradients for batch
      update_weights learning_rate, momentum

      # return average error for batch
      total_error / (data.length.to_f)
    end

    def calculate_training_error ideal_output
      @outputs[@output_layer].map.with_index do |output, i| 
        output - ideal_output[i]
      end
    end

    # Propagate the training error backwards through the network
    def calculate_deltas training_error
      @deltas = []

      @output_layer.downto(1).each do |layer|
        @deltas[layer] = []

        target_layer = layer + 1 # i.e. the layer that feeds into this one, when working backwards
        target_deltas = @deltas[target_layer]
        target_weights = @weights[target_layer]

        @shape[layer].times do |neuron|
          output = @outputs[layer][neuron]
          activation_derivative = output * (1.0 - output)

          @deltas[layer][neuron] = if layer == @output_layer
            # For neurons in output layer, use training error
            -training_error[neuron] * activation_derivative
          else
            # For neurons in hidden layers, weight deltas from target layer
            weighted_target_deltas = target_deltas.map.with_index do |target_delta, target_neuron| 
              target_weight = target_weights[target_neuron][neuron]
              target_delta * target_weight
            end

            sum_of_weighted_target_deltas = weighted_target_deltas.reduce(:+)
            activation_derivative * sum_of_weighted_target_deltas
          end
        end
      end
    end

    # After backpropagating the error, we can move forward again to calculate gradients
    # For a batch of training data, we accumulate the gradients, so that we can then update weights once for the batch
    def calculate_gradients
      1.upto(@output_layer).each do |layer|
        source_layer = layer - 1 # i.e, the layer that is feeding into this one
        source_neurons = @shape[source_layer] + 1 # account for bias neuron

        @shape[layer].times do |neuron|
          delta = @deltas[layer][neuron]

          source_neurons.times do |source_neuron|
            source_output = @outputs[source_layer][source_neuron] || 1 # if no output, this is the bias neuron
            gradient = source_output * delta
            @gradients[layer][neuron][source_neuron] += gradient # accumulate gradients from batch
          end
        end
      end
    end

    # Now that we've calculated gradients for the batch, we can use these to update the weights
    def update_weights learning_rate, momentum
      1.upto(@output_layer) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # account for bias neuron

        @shape[layer].times do |neuron|
          source_neurons.times do |source_neuron|
            gradient = @gradients[layer][neuron][source_neuron]
            previous_weight_change = @weight_changes[layer][neuron][source_neuron]

            weight_change = (learning_rate * gradient) + (momentum * previous_weight_change)

            @weights[layer][neuron][source_neuron] += weight_change
            @weight_changes[layer][neuron][source_neuron] = weight_change
          end
        end
      end
    end

    def set_weight_changes_to_zeros
      @weight_changes = build_matrix { 0.0 }
    end

    def set_gradients_to_zeroes
      @gradients = build_matrix { 0.0 }
    end

    def set_random_weight_values
      @weights = build_matrix { rand(-1.0..1.0) }    
    end

    def build_matrix
      Array.new(@shape.length) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # account for bias neuron

        if source_layer >= 0
          Array.new(@shape[layer]) do |neuron|
            Array.new(source_neurons) { yield }
          end
        end
      end
    end

    # http://en.wikipedia.org/wiki/Sigmoid_function
    def sigmoid x
      1 / (1 + Math::E**-x)
    end

    # http://en.wikipedia.org/wiki/Mean_squared_error
    def mean_squared_error errors
      errors.map {|e| e**2}.reduce(:+) / errors.length.to_f
    end
end
