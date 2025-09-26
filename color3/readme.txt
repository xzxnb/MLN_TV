~edge(X,X).
edge(X,Y) -> edge(Y,X).
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).
edge(X,Y) -> ~(red(X) & red(Y)) & ~(blue(X) & blue(Y)) & ~(green(X) & green(Y)).

red_blue(X,Y) <-> edge(X,Y) & red(X) & blue(Y).
red_green(X,Y) <-> edge(X,Y) & red(X) & green(Y).
green_blue(X,Y) <-> edge(X,Y) & green(X) & blue(Y).
V = 20
|red_blue| >= m1
|red_green| >= m1
|green_blue| >= m1


~edge(X,X).
edge(X,Y) -> edge(Y,X).
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).
edge(X,Y) -> ~(red(X) & red(Y)) & ~(blue(X) & blue(Y)) & ~(green(X) & green(Y)).

red_blue(X,Y) <-> edge(X,Y) & red(X) & blue(Y).
red_green(X,Y) <-> edge(X,Y) & red(X) & green(Y).
green_blue(X,Y) <-> edge(X,Y) & green(X) & blue(Y).
V = 20
|red_blue| >= m2
|red_green| >= m2
|green_blue| >= m2


These three tables represent the TV-distance of the above two mlns when domain is 10, 15, and 20 respectively.

