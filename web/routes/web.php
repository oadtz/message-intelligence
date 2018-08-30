<?php


Route::prefix('api')->group(function () {
    Route::get('flights', 'Api\FlightController@getFlights');
    Route::post('flights', 'Api\FlightController@addFlight');
    Route::any('chatbot', 'Api\ChatbotController@chat');
});

Route::get('test', function () {
    $price = $_GET['price'];
    $discount = $_GET['discount'];
    $saving = $price * ($discount / 100.0);


    dd(config('app.model_api_url'));
});

Route::get('/flights', 'FlightController@index');
Route::get('/chatbot', 'SiteController@chatbot');
Route::get('/', 'SiteController@index');
