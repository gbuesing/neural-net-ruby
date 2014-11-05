class NeuralNet
  attr_reader :shape, :weights, :outputs

  DEFAULT_TRAINING_OPTIONS = {
    max_iterations:   20_000,
    learning_rate:    0.3,
    momentum:         0.1,
    error_threshold:  0.05
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
    # input to this method represents the output of the first layer (i.e., the input layer)
    @outputs = [input]

    # calculate output of subsequent layers
    1.upto(@output_layer).each do |layer|
      source_layer = layer - 1
      source_outputs = @outputs[source_layer]

      @outputs[layer] = @weights[layer].map do |neuron_weights|
        weighted_inputs = neuron_weights.map.with_index do |weight, i| 
          input = source_outputs[i] || 1 # assume bias neuron if no input
          weight * input
        end

        sigmoid weighted_inputs.reduce(:+)
      end
    end

    # Output of last layer is the final result
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

    def calculate_deltas training_error
      @deltas = []
      # Calculation of node delta for non-output layers requires the node delta of its target layer
      # therefore, we walk backwards through layers
      # Stop at 1; no need to calculate for input layer
      @output_layer.downto(1).each do |layer|
        @deltas[layer] = []
        is_output_layer = layer == @output_layer

        target_layer = layer + 1
        target_deltas = @deltas[target_layer]
        target_weights = @weights[target_layer]

        @shape[layer].times do |neuron|
          output = @outputs[layer][neuron]
          derivative = output * (1.0 - output)

          @deltas[layer][neuron] = if is_output_layer
            -training_error[neuron] * derivative
          else
            weighted_target_deltas = target_deltas.map.with_index do |delta, target_neuron| 
              delta * target_weights[target_neuron][neuron]
            end

            derivative * weighted_target_deltas.reduce(:+)
          end
        end
      end
    end

    def calculate_gradients
      1.upto(@output_layer).each do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # include extra bias neuron

        @shape[layer].times do |neuron|
          delta = @deltas[layer][neuron]

          source_neurons.times do |source_neuron|
            output = @outputs[source_layer][source_neuron] || 1 # if no output, assume bias neuron
            @gradients[layer][neuron][source_neuron] += output * delta # accumulate gradients from batch
          end
        end
      end
    end

    def update_weights learning_rate, momentum
      1.upto(@output_layer) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # include bias neuron

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
        source_neurons = @shape[source_layer] + 1 # include extra bias neuron

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
