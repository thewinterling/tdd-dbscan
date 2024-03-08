# Example repository for a test driven development approach
Example repository for a test driven development approach 
on the example of an implementation of a dbscan implementation.

The goal of this repository was to create a small
example for how one can use test driven development to
implement clean and testable code. 
And it was also a test to see if the implementation using
such an approach really takes more time than the
"normal" approach of directly start with the implementation.

## Implementation

The algorightm was implemented by leveraging the pseudo code
from [the wikipedia page](https://en.wikipedia.org/wiki/DBSCAN).

We started with the simple test for the `Point` class with its 
neighboorhood method. The tests are name such that the test itself
do not need additional comments, since the tests itself are already
self-explainable - and in this example - also quite short.
We defined the (wanted) behaviour of the class through these tests.
Once we wrote the tests, we started the implementation - and thus 
went from failing to passing tests.

We continued for the `Cluster` class and the slightly more complex
class `Dbscan` in the same way. Writing tests to define _what_ we 
expect from the implementation, then _do_ the actual implementation work.

The last test was to check the overall behaviour of the algorithm
by checking the output of our implementation against the 
`sklearn.cluster.DBSCAN` output.
