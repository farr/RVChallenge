module Periodogram

export periodogram

""" Returns a maximum-likelihood estimate of the amplitude of a
sinusoidal signal in the series of `xs`, sampled at times `ts`, for
each of the frequencies `fs`. """
function periodogram(ts, xs, fs)
    amps = zeros(fs)

    for i in eachindex(amps)
        cs = cos(2*pi*fs[i]*ts)
        ss = sin(2*pi*fs[i]*ts)
        os = ones(ts)

        A = cat(2, cs, ss, os)
        a,b,m = A\xs

        amps[i] = sqrt(a*a + b*b)
    end

    amps
end

end
