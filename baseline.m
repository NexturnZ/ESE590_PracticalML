function z = baseline(data, lambda, p)

m = length(data);
D = diff(speye(m),2);
w = ones(m,1);
for it = 1:10
   W = spdiags(w,0,m,m);
   C = chol(W+lambda*D'*D);
   z = C\(C'\(w.*data));
   w = p*(data>z)+(1-p)*(data<z);
end


