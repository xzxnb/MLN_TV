k_color1.mln :
~edge(X,X).
edge(X,Y) -> edge(Y,X).
1.3 edge(X,Y) -> ~(red(X) & red(Y)) & ~(blue(X) & blue(Y)) & ~(green(X) & green(Y))
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).

k_color2.mln :
~edge(X,X).
edge(X,Y) -> edge(Y,X).
1.3 \exists Y: (edge(X,Y) & blue(Y))
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).

Each excel file name in this folder indicates which mln is used and the size of the corresponding domain. Each excel file contains 3 columns of data, and each row represents the TV-distance under two different weights. The following is an example:
The data in row 16 of the k_color1_domain10.xlsx file is 0.1, 1.5, 1, which means that the TV-distance of the following two mlns is 1:
~edge(X,X).
edge(X,Y) -> edge(Y,X).
0.1 edge(X,Y) -> ~(red(X) & red(Y)) & ~(blue(X) & blue(Y)) & ~(green(X) & green(Y))
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).
V =10


~edge(X,X).
edge(X,Y) -> edge(Y,X).
1.5 edge(X,Y) -> ~(red(X) & red(Y)) & ~(blue(X) & blue(Y)) & ~(green(X) & green(Y))
(red(X) & ~blue(X) & ~green(X)) | (~red(X) & blue(X) & ~green(X)) | (~red(X) & ~blue(X) & green(X)).
V =10



