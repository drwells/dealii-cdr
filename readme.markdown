dealii-cdr
==========
This repository contains several related projects. My goal here is to attempt to
understand adaptive mesh refinement and parallel computing with `deal.II` by
solving a sequence of convection-diffusion-reaction problems. In particular, I
start with a 'vanilla' solver that does not contain any parallelism or
adaptivity and slowly 'locally refine' (ha ha) this example to contain more
interesting features.


Why use convection-diffusion-reaction?
--------------------------------------
This problem exhibits very fine boundary layers (usually, from the literature,
these layers have width equal to the square root of the diffusion coefficient on
internal layers and the diffusion coefficient on boundary layers). A good way to
solve it is to use adaptive mesh refinement to refine the mesh only at interior
boundary layers. At the same time, this problem is linear (and does not have a
pressure term) so it is much simpler to solve than the Navier-Stokes equations
with comparable diffusion (Reynolds number).

Why use so many sample problems?
--------------------------------
John Gall wrote "A complex system that works is invariably found to have evolved
from a simple system that worked. A complex system designed from scratch never
works and cannot be patched up to make it work. You have to start over with a
working simple system." I believe this usually holds for software. I suspect
I would fail miserably if I sat down and tried to write a distributed memory
Navier-Stokes solver with all of the bells and whistles.

Of course, this approach may backfire: If you are a fan of horror stories then I
recommend
http://www.tomdalling.com/blog/software-design/fizzbuzz-in-too-much-detail/(this
frightening tale).

Requirements
------------
* A `C++-14` compliant compiler (I use a lot of `unique_ptr`s and `constexpr`s)
* A recent (8.2 or newer) version of `deal.II`

Recommended Literature
----------------------
I need to figure out how to handle citations with markdown. This list will grow
a lot as this project progresses.
* "Robust Numerical Methods for Singularly Perturbed Problems" by Roos, Stynes,
  and Tobiska.
