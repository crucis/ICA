function [ y, B ] = EASI( x, mu, T, nsources, nsensors, B )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 6
        B = randn(nsources, nsensors);
    end
    for t=1:T

        y = B*x;
        y2 = y*y';
    
        Identity = eye(nsensors);
        %g	= diag(y2).*y		;
        %g = y .*sqrt(diag(y2));
        g = tanh(y);
        %gy = g*y';
        
        G = y2 - Identity + g*y' - y*g';
        dB = -mu*G*B;
        
        B = B+dB;
        if (any(~isfinite(B)))
            warning('Lost convergence at iterator %i; lower learning rate?', t);
            break;
        end
    end
    y = B*x;
end

