function [start,stop] = len_peak(data,peak_loc,bl)

start = peak_loc;
while data(start)>bl(start)
    start = start-1;
    if start<1
        break
    end
end

stop = peak_loc;
while data(stop)>bl(stop)
    stop = stop+1;
    if stop>length(data)
        break
    end
end
