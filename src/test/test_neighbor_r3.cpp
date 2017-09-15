#include "catch.hpp"

#define N_RAD 3
#define POP_WIDTH 10

#include "../util/neighbor.cu"

TEST_CASE( "neighbors are calculated correctly (N_RAD 3)", "[neighbor]" ) {
  REQUIRE( get_neighbor(5, 5, 0) == 24 );
  REQUIRE( get_neighbor(5, 5, 6) == 30 );
  REQUIRE( get_neighbor(5, 5, 24) == 60 );
  REQUIRE( get_neighbor(5, 5, 42) == 90 );
  REQUIRE( get_neighbor(5, 5, 48) == 96 );
}
