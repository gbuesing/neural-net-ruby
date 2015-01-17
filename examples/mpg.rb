#!/usr/bin/env ruby

require_relative '../neural_net'
require 'csv'

# Dataset available at https://archive.ics.uci.edu/ml/datasets/Auto+MPG
rows = CSV.read "examples/auto-mpg.data", col_sep: ' '

rows.shuffle!

normalize = -> (val, fromLow, fromHigh, toLow, toHigh) {  (val - fromLow) * (toHigh - toLow) / (fromHigh - fromLow).to_f }


x_data, y_data = [], []

rows.each do |row|
  mpg = normalize.(row[0].to_f, 0, 60, 0, 1)
  cylinders = normalize.(row[1].to_f, 2, 12, 0, 1)
  displacement = normalize.(row[2].to_f, 10, 1000, 0, 1)
  horsepower = normalize.(row[3].to_f, 1, 1000, 0, 1)
  weight = normalize.(row[4].to_f, 100, 5000, 0, 1)

  x_data << [cylinders, displacement, horsepower, weight]
  y_data << [mpg]
end

test_size = 100
train_size = rows.length - test_size

x_train = x_data.slice(0, train_size)
y_train = y_data.slice(0, train_size)
x_test = x_data.slice(train_size, test_size)
y_test = y_data.slice(train_size, test_size)

to_mpg = -> (flt) { normalize.(flt, 0, 1, 0, 60) }

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

run_test = -> (nn, inputs, expected_outputs) {
  mpg_err, errsum = 0, 0
  outputs = []

  inputs.each.with_index do |input, i|
    output = nn.run input
    outputs << output
    mpg_err += (to_mpg.(output[0]) - to_mpg.(expected_outputs[i][0])).abs
    errsum += mse.(output, expected_outputs[i])
  end

  y_mean = expected_outputs.inject(0.0) { |sum, val| sum + val[0] } / expected_outputs.size
  y_sum_squares = expected_outputs.map{|val| (val[0] - y_mean)**2 }.reduce(:+)
  y_residual_sum_squares = outputs.zip(expected_outputs).map {|out, expected| (expected[0] - out[0])**2 }.reduce(:+)
  r_squared = 1.0 - (y_residual_sum_squares / y_sum_squares)

  [mpg_err / inputs.length.to_f, errsum / inputs.length.to_f, r_squared]
}

show_examples = -> (nn, x, y) {
  puts "Actual\tPredict\tError (mpg)"
  10.times do |i|
    output = nn.run x[i]
    predicted = to_mpg.(output[0])
    actual = to_mpg.(y[i][0])
    puts "#{actual.round(1)}\t#{predicted.round(1)}\t#{(predicted - actual).abs.round(1)}"
  end
}

nn = NeuralNet.new [4, 4, 1]

puts "Testing the untrained network..."
mpg_err, avg_mse, r_squared = run_test.(nn, x_test, y_test)
puts "Average prediction error: #{mpg_err.round(2)} mpg (mse: #{(avg_mse * 100).round(2)}%, r-squared: #{r_squared.round(2)})"

# puts "\nUntrained test examples (first 10):"
# show_examples.(nn, x_test, y_test)

puts "\nTraining the network...\n\n"
t1 = Time.now
result = nn.train(x_train, y_train, error_threshold: 0.005, 
                                    max_iterations: 100,
                                    log_every: 10
                                    )

# puts result
puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).round(1)}s"

puts "\nTesting the trained network..."
mpg_err, avg_mse, r_squared = run_test.(nn, x_test, y_test)
puts "Average prediction error: #{mpg_err.round(2)} mpg (mse: #{(avg_mse * 100).round(2)}%, r-squared: #{r_squared.round(2)})"

puts "\nTrained test examples (first 10):"
show_examples.(nn, x_test, y_test)
