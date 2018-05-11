using Distributions, PyPlot, Colors

# a car struct declaration with its atributes
mutable struct car{TMV <:Integer, TA <:Integer, TCV <:Integer, TL <:Integer, TD <:Integer}
    max_v::TMV # maximum velocity of a car (constant)
    acceleration::TA # acceleration of a car (constant)
    curr_v::TCV # velocity in current iteration
    line::TL # line on which the car is moving in current iteration
    distance::TD # the distance from the road beginning
end

# function looks for the next car on the given line. It returns its current position (distance)
function next_car(line, position, road, dim)
  position += 1
  x = 0
  y = 0
  while x !=line
      loc = findnext(v -> v != -50, road, position)
      if loc == 0
          return dim * 10
      end
      x, y = ind2sub(road,loc)
      position = loc + 1
  end
  return y
end

# function looks for the previous car on the givem line. It returns its current position after adding its current velocity
function previous_car(line, position, road)
  position -= 1
  x = 0
  y = 0
  while x !=line
      loc = findprev(v -> v != -50, road, position)
      if loc == 0
          return 0
      end
      x, y = ind2sub(road,loc)
      position = loc - 1
  end
  y + road[x,y]
end

#function that populates the road
function fill_road(road, line, distance, value)
  for i in 0:4
      road[line, distance - i] = value
  end
end

#function that creates the cars at the beggining of the road
function create_new_cars(n_lines, road, dim, Cars, speed_limit)
  for i in 1:n_lines
      road[i,5] != -50 && continue
      max_v =  round(Int, rand(Normal(speed_limit, 2)))
      curr_v = floor(Int, 0.5*max_v)
      fill_road(road, i, 5, curr_v)
      acceleration = rand(1:4)
      c = car(max_v, acceleration, curr_v, i, 5)
      Cars = vcat(Cars, c)
  end
  return Cars
end

#= overtaking (changing the line):
      - a car checks the distance to the next car on the current line and lines one up and one down
      - iteratively for the lines with the biggest distance to the next car it check whether it can safely change the line (whether the previous car on a given line is far enough)
      - if a car can safely change the line, the car's parameter "line" changes instantly
The function changes the parameter "line" of a car and returns the distnace to the next car on the new line =#
function change_line(road, car, position, next_car_pos, space, dim, n_lines)
  lines = Dict(car.line => next_car_pos)
  for i in (-1, 1)
      (car.line + i < 1 || car.line + i > n_lines) && continue
      lines[car.line + i] = next_car(car.line + i, position, road, dim)
  end
  while true
      line = collect(keys(lines))[indmax(collect(values(lines)))]
      car.line == line && return space
      prev_car = previous_car(line, position, road)
      if prev_car >= car.distance
          delete!(lines, line)
      else
          fill_road(road, car.line, car.distance, -50)
          car.line = line
          fill_road(road, car.line, car.distance, car.curr_v)
          break
      end
  end
  new_next_car_pos = maximum(values(lines))
  return new_next_car_pos - car.distance - 1
end

# for every car the function calculates the velocity in a given iteration and decides whether the car wants to change the line
  for car in Cars
      new_v = min(car.curr_v + car.acceleration, car.max_v)
      position = sub2ind(road, car.line, car.distance)
      next_car_pos = next_car(car.line, position, road, dim)
      space = next_car_pos - car.distance - 1
      if space < new_v
          space = change_line(road, car, position, next_car_pos, space, dim, n_lines)
      end
      new_v = min(space, new_v)
      rand() < slowing_parameter && (new_v = max(0, new_v - 1))
      car.curr_v = new_v
  end
end

# function that moves all the cars on the road forward by set velocity value
function move(Cars, road, dim)
  for car in Cars
      fill_road(road, car.line, car.distance, -50)
      car.distance = car.distance + car.curr_v
      car.distance > dim && continue
      fill_road(road, car.line, car.distance, car.curr_v)
  end
end

#visualization
function visualise(iterations, n_lines, dim, slowing_parameter, speed_limit)
  Cars = Any[]
  road = fill(-50, n_lines, dim)
  road[1,1] = 20
  img = imshow(road, aspect=4)
  road[1,1] = -50
  for i in 1:iterations
      Cars = create_new_cars(n_lines, road, dim, Cars, speed_limit)
      sort(Cars, by = x -> x.distance, rev=true)
      set_velocity(Cars, road, dim, slowing_parameter, n_lines)
      move(Cars, road, dim)
      filter!(x -> x.distance <= dim, Cars)
      img[:set_data](road)
      show()
      sleep(0.2)
  end
end

visualise(
            iterations = 300, #number of simulation iterations
            n_lines = 3, # how many lines the highway has
            dim = 1000, # how long is the simulated part of highway (in meters)
            slowing_parameter = 0.2, # probability that a car will slow down by one in every iteration (from Nagel Schreck model)
            speed_limit = 15 # the speed limit on the road (in meters/second)
            )
