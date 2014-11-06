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


# Normalize data values before feeding into network
normalize = -> (val, high, low) {  (val - low) / (high - low) } # maps input to float between 0 and 1

columns = (0..3).map do |i|
  iris_data.map {|row| row[0][i] }
end

iris_data.each.with_index do |row, i|
  normalized = row[0].map.with_index do |val, j| 
    max, min = columns[j].max, columns[j].min
    normalize.(val, max, min)
  end
  iris_data[i][0] = normalized
end


iris_data.shuffle!
train_data = iris_data.slice(0, 100)
test_data = iris_data.slice(100, 50)

# Build a 3 layer network: 4 input neurons, 4 hidden neurons, 3 output neurons
# Bias neurons are automatically added to input + hidden layers; no need to specify these
nn = NeuralNet.new [4,4,3]

puts "Testing the untrained network..."

prediction_success = -> (actual, ideal) {
  predicted = (0..2).max_by {|i| actual[i] }
  ideal[predicted] == 1 
}

success, failure = 0,0
test_data.each do |input, expected|
  output = nn.run input
  prediction_success.(output, expected) ? success += 1 : failure += 1
end

puts "Untrained prediction success: #{success}, failure: #{failure}"


puts "\nTraining the network...\n\n"

t1 = Time.now
result = nn.train(train_data, learning_rate: 0.3, 
                              momentum: 0.1,
                              error_threshold: 0.005, 
                              max_iterations: 2_000,
                              log_every: 100
                              )

# puts result
puts "\nDone training the network. #{(Time.now - t1).round(1)}s"


puts "\nTesting the trained network..."

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

success, failure, errorsum = 0,0,0

test_data.each do |input, expected|
  output = nn.run input
  prediction_success.(output, expected) ? success += 1 : failure += 1
  errorsum += mse.(output, expected)
end

puts "Trained prediction success: #{success}, failure: #{failure}"
# puts "Test data error: #{(errorsum / test_data.length.to_f).round(5)}"
