class NeuralNet
  attr_reader :shape, :outputs
  attr_accessor :weights, :weight_update_values

  DEFAULT_TRAINING_OPTIONS = {
    max_iterations:   1_000,
    error_threshold:  0.01
  }

  def initialize(shape)
    @shape = shape
    @output_layer = @shape.length - 1
    set_initial_weight_values
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

  def train data, opts = {}
    opts = DEFAULT_TRAINING_OPTIONS.merge(opts)
    error_threshold, log_every = opts[:error_threshold], opts[:log_every]
    iteration = 0
    error = nil

    set_weight_changes_to_zeros
    set_initial_weight_update_values if @weight_update_values.nil?
    set_previous_gradients_to_zeroes

    while iteration < opts[:max_iterations]
      iteration += 1

      error = train_on_batch(data)
      
      if log_every && (iteration % log_every == 0)
        puts "[#{iteration}] error: #{error.round(5)}"
      end

      break if error_threshold && (error < error_threshold)
    end

    {error: error.round(5), iterations: iteration, below_error_threshold: (error < error_threshold)}
  end

  private

    def train_on_batch data
      total_mse = 0

      set_gradients_to_zeroes

      data.each do |(input, ideal_output)|
        run input
        training_error = calculate_training_error ideal_output
        update_gradients training_error
        total_mse += mean_squared_error training_error
      end

      update_weights

      total_mse / data.length.to_f # average mean squared error for batch
    end

    def calculate_training_error ideal_output
      @outputs[@output_layer].map.with_index do |output, i| 
        output - ideal_output[i]
      end
    end

    def update_gradients training_error
      deltas = []
      # Starting from output layer and working backwards, backpropagating the training error
      @output_layer.downto(1).each do |layer|
        deltas[layer] = []
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # account for bias neuron
        target_layer = layer + 1

        @shape[layer].times do |neuron|
          output = @outputs[layer][neuron]
          activation_derivative = output * (1.0 - output)

          # calculate delta for neuron
          delta = deltas[layer][neuron] = if layer == @output_layer
            # For neurons in output layer, use training error
            -training_error[neuron] * activation_derivative
          else
            # For neurons in hidden layers, weight deltas from target layer
            weighted_target_deltas = deltas[target_layer].map.with_index do |target_delta, target_neuron| 
              target_weight = @weights[target_layer][target_neuron][neuron]
              target_delta * target_weight
            end

            sum_of_weighted_target_deltas = weighted_target_deltas.reduce(:+)
            activation_derivative * sum_of_weighted_target_deltas
          end

          # use delta to calculate gradients
          source_neurons.times do |source_neuron|
            source_output = @outputs[source_layer][source_neuron] || 1 # if no output, this is the bias neuron
            gradient = source_output * delta
            @gradients[layer][neuron][source_neuron] += gradient # accumulate gradients from batch
          end
        end
      end
    end

    MIN_STEP, MAX_STEP = Math::E**-6, 50

    # Now that we've calculated gradients for the batch, we can use these to update the weights
    # Using the RPROP algorithm - somewhat more complicated than classic backpropagation algorithm, but much faster
    def update_weights
      1.upto(@output_layer) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # account for bias neuron

        @shape[layer].times do |neuron|
          source_neurons.times do |source_neuron|
            weight = @weights[layer][neuron][source_neuron]
            weight_change = @weight_changes[layer][neuron][source_neuron]
            weight_update_value = @weight_update_values[layer][neuron][source_neuron]
            # for RPROP, we use the negative of the calculated gradient
            gradient = -@gradients[layer][neuron][source_neuron]
            previous_gradient = @previous_gradients[layer][neuron][source_neuron]

            c = sign(gradient * previous_gradient)

            case c
              when 1 then # no sign change; accelerate gradient descent
                weight_update_value = [weight_update_value * 1.2, MAX_STEP].min
                weight_change = -sign(gradient) * weight_update_value
              when -1 then # sign change; we've jumped over a local minimum
                weight_update_value = [weight_update_value * 0.5, MIN_STEP].max
                weight_change = -weight_change # roll back previous weight change
                gradient = 0 # so won't trigger sign change on next update
              when 0 then
                weight_change = -sign(gradient) * weight_update_value
            end

            @weights[layer][neuron][source_neuron] += weight_change
            @weight_changes[layer][neuron][source_neuron] = weight_change
            @weight_update_values[layer][neuron][source_neuron] = weight_update_value
            @previous_gradients[layer][neuron][source_neuron] = gradient
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

    def set_previous_gradients_to_zeroes
      @previous_gradients = build_matrix { 0.0 }
    end

    def set_initial_weight_update_values
      @weight_update_values = build_matrix { 0.1 }
    end

    def set_initial_weight_values
      # Initialize all weights to random float value
      @weights = build_matrix { rand(-0.5..0.5) }  

      # Update weights for first hidden layer (Nguyen-Widrow method)
      # This is a bit obscure, and not entirely necessary, but it should help the network train faster
      beta = 0.7 * @shape[1]**(1.0 / @shape[0])

      @shape[1].times do |neuron|
        weights = @weights[1][neuron]
        norm = Math.sqrt weights.map {|w| w**2}.reduce(:+)
        updated_weights = weights.map {|weight| (beta * weight) / norm }
        @weights[1][neuron] = updated_weights
      end
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

    def sigmoid x
      1 / (1 + Math::E**-x)
    end

    def mean_squared_error errors
      errors.map {|e| e**2}.reduce(:+) / errors.length.to_f
    end

    ZERO_TOLERANCE = Math::E**-16

    def sign x
      if x < ZERO_TOLERANCE && x > -ZERO_TOLERANCE # if float very close to 0
        0
      else
        x <=> 0 # returns 1 if postitive, -1 if negative
      end
    end
end
