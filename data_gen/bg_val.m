function y = bg_val(x, c, sigma, min_speed, max_speed)
    z = (x - c)/sigma;
    t = exp(-vecnorm(z, 2, 2).^2);
    y = t*(max_speed - min_speed) + min_speed;
    
end
