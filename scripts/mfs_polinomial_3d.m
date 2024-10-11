function main()
  N = 4
  global A; A = zeros(N,N)
  global Beta; Beta=1
  global nu; nu =1


  for nn = 0:N
    for mm = 0:nn
      printf('%i:%i\n',nn, mm)
      polinomial_3d(nn,mm)
    endfor
  ##  printf('%i\n',nn)
  endfor

endfunction


function polinomial_3d(n,k)
  global A
  global Beta
  global nu
  if k == 0
    A(n+1,k+1) = 1
    return
  endif
  if n == k
    A(n+1,n+1) = -(A(n,n)*Beta)/(2*nu*n)
    return
  endif
  res = k*(k+1)*A(n,k+2)
  for m = (k-1):(n-1)
    res -= Beta*A(m+1,k)
  endfor
  res *= (1/(2*nu*k))
  A(n+1,k+1) = res
end

##function polinomial_3d(n,k)
##  global A
##  global Beta
##  global nu
##  if k == 0
##    A(n,k) = 1
##    return
##  endif
##  if n == k
##    A(n,n) = -(A(n-1,n-1)*Beta)/(2*nu*n)
##    return
##  endif
##  res = k*(k+1)*A(n,k+1)
##  for m = (k-1):(n-1)
##    res -= Beta*A(m,k-1)
##  endfor
##  res *= (1/(2*nu*k))
##  A(n,k) = res
##end







