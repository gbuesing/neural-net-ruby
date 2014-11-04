require_relative 'neural_net'

# This neural network will predict the species of an iris based on sepal and petal size
# Dataset: http://en.wikipedia.org/wiki/Iris_flower_data_set

label_encodings = {
  "Iris-setosa"     => [1, 0, 0], 
  "Iris-versicolor" => [0, 1, 0], 
  "Iris-virginica"  => [0, 0 ,1]
}

iris_data = []

File.open("iris.data") do |f|
  while line = f.gets
    values = line.chomp.split(',')
    label = values.pop
    iris_data << [ values.map(&:to_f), label_encodings[label] ]
  end
end

iris_data.shuffle!
train_data = iris_data.slice(0, 100)
test_data = iris_data.slice(100, 50)


nn = NeuralNet.new [4,4,3]

# 1. Train the network...

result = nn.train(train_data, learning_rate: 0.05, 
                              momentum: 0.01,
                              error_threshold: 0.005, 
                              max_iterations: 2_000,
                              log_every: 100
                              )

puts result


# 2. Test the trained network...

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

success, failure, errorsum = 0,0,0

test_data.each do |input, expected|
  output = nn.run input

  errorsum += mse.(output, expected)

  # highest output value indicates predicted label for input
  predicted = (0..2).max_by {|i| output[i] }
  prediction_success = expected[predicted] == 1 

  prediction_success ? success += 1 : failure += 1
end

puts "Test data error: #{(errorsum / test_data.length.to_f).round(5)}"
puts "Test prediction success: #{success}, failure: #{failure}"
