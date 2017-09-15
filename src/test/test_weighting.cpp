#include "catch.hpp"
#include <random>

#define POP_WIDTH 100

#include "../util/weighting.cu"

TEST_CASE( "multiplication of resulting weigths to objectives", "[weighting]" ) {

  
  SECTION( "no offset" ) {
    float obj[] = { 1.0, 2.0, 3.0 };
    float weights[] = { 4.0, 5.5, 6.3 };

    REQUIRE( weight_multiply( obj, weights, 1 ) == Approx( 33.9 ) );
  }
  SECTION( "offset == 3" ) {
    float obj[] = { 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0 };
    float weights[] = { 4.0, 5.5, 6.3 };

    REQUIRE( weight_multiply( obj, weights, 3 ) == Approx( 33.9 ) );
  }
}

TEST_CASE( "weight calculation (POP = 100)", "[weighting]" ) {

  SECTION( "no offset" ) {
    float res_weights[3];

    // the 3 edges of the upper left triangle
    calc_weights( 0, 0, res_weights, 1 );
    REQUIRE( res_weights[0] == Approx( 1.0 ) );
    REQUIRE( res_weights[1] == Approx( 0.0 ) );
    REQUIRE( res_weights[2] == Approx( 0.0 ) );
    
    calc_weights( 99, 0, res_weights, 1 );
    REQUIRE( res_weights[0] == Approx( 0.005050441 ) );
    REQUIRE( res_weights[1] == Approx( 0.999987246 ) );
    REQUIRE( res_weights[2] == Approx( 0.0 ) );

    calc_weights( 0, 99, res_weights, 1 );
    REQUIRE( res_weights[0] == Approx( 0.005050441 ) );
    REQUIRE( res_weights[1] == Approx( 0.0 ) );
    REQUIRE( res_weights[2] == Approx( 0.999987246 ) );

    // a point in the middle
    // 0,075376884
    // 0,341708543
    // 0,582914573
    // 0,679879256
    calc_weights( 34, 58, res_weights, 1 );
    REQUIRE( res_weights[0] == Approx( 0.110868045 ) );
    REQUIRE( res_weights[1] == Approx( 0.502601808 ) );
    REQUIRE( res_weights[2] == Approx( 0.857379554 ) );

    // points on the mirrored side
  }
}
