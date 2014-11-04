require_relative 'neural_net'

# XOR solving is uninteresting, but it's the hello world of neural networks...

nn = NeuralNet.new [2,2,1]
# nn.weights = [nil, [[-0.07, 0.22, -0.46], [0.94, 0.46, 0.10]], [[-0.22, 0.58, 0.78]]]

training_data = [
  [ [1, 0], [1] ],
  [ [0, 0], [0] ],
  [ [0, 1], [1] ],
  [ [1, 1], [0] ]
]

# 1. Train the network...

result = nn.train(training_data,  learning_rate: 0.7, 
                                  momentum: 0.3,
                                  max_iterations: 1_000, 
                                  error_threshold: 0.005, 
                                  log_every: 100
                                  )

puts result


# 2. Test the trained network...

success, failure = 0, 0

training_data.each do |(input, expected)|
  output = nn.run input

  if output[0].round == expected[0]
    success += 1
  else
    failure += 1
  end

  puts "#{input.inspect}: [#{output[0].round(2)}] #{(output[0] > 0.5).inspect}"  
end

puts "Test prediction success: #{success}, failure: #{failure}"
